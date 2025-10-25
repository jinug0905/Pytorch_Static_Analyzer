// module load GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.7.0 CUDA/12.9.0

#include <torch/script.h>

// headers for passes on 2.7.0
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/freeze_module.h>


#include <ATen/ATen.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <sstream>

// ----- Helpers -----
static size_t scalar_type_size(c10::ScalarType t) {
  return c10::elementSize(t); // bytes per element
}

// tensor shape and dtype => # of bytes
static size_t tensor_type_nbytes(const c10::VaryingShape<int64_t>& sizes, c10::ScalarType st) {
  auto cs = sizes.concrete_sizes();
  if (!cs.has_value()) return 0;

  const auto& dims = cs.value(); // std::vector<int64_t>
  int64_t n = 1;
  for (auto d : dims)
    n *= d;

  return static_cast<size_t>(n) * scalar_type_size(st);
}


// How to display in recognizable form
static std::string pretty(size_t n) {
  const char* units[] = {"B","KB","MB","GB","TB"};

  double d = static_cast<double>(n);
  int u = 0;
  while (d >= 1024.0 && u < 4) {
    d /= 1024.0;
    ++u;
  }

  std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(5);
  oss << d << " " << units[u];
  return oss.str();
}


/*
  Compute liveness of tensor : where does a tensor die???
*/
static std::unordered_map<const torch::jit::Value*, size_t> compute_last_use_indices(const std::vector<torch::jit::Node*>& topo, const std::unordered_map<torch::jit::Node*, size_t>& nodeIdx) {
  std::unordered_map<const torch::jit::Value*, size_t> lastUse;
  for (size_t i = 0; i < topo.size(); ++i) {

    auto* node = topo[i];
    for (auto* val : node->outputs()) {

      if (val->uses().empty()) {
        lastUse[val] = i;

      } else {

        size_t maxIdx = i;
        for (const auto& usg : val->uses()) {

          auto it = nodeIdx.find(usg.user);
          if (it != nodeIdx.end() && it->second > maxIdx)
            maxIdx = it->second;
        }

        lastUse[val] = maxIdx;
      }
    }
  }

  return lastUse;
}

/*
  PyTorch’s memory management's 2 main categories of allocations:
  ----------------------------------------------------------------------------------
  Persistent: Model parameters and buffers

    - Allocated once when the model is built (or loaded)  
    - Always live (until you delete or move the module),

  Transient : Activations, temporary tensors, intermediate outputs

    - Allocated during the forward/backward pass, released when no longer needed
      - Freed logically when the tensor is destroyed,
        but the physical GPU memory is cached for reuse.
    - Follows tensor liveness determined by Python scope and autograd graph
*/

int main() {
  try {

    /*
      load TorchScript module from .pt file
      it contains
        - method bytecode/IR (e.g. forward)
        - the module hierarchy and attributes
        - parameters/buffers
          - parameters : this is learnable components in NN
            - requires_grad=True is default
            - ex : usually means weights and biases.
          - buffers : these are tensor values that are associated with a module's state but not learnable
            - requires_grad=False is default
            - ex : running mean and variance in a batch normalization layer in nn.BatchNorm2d

      module.get_method("forward").graph() -> give high-level graph for forward pass

      method.graph() -> Graph object describe IR
        - nodes (ops like aten::linear, aten::relu, prim::GetAttr)
        - inputs/outputs (graph->inputs(), graph->outputs())
        - SSA values (Value*) and their Types
    */

    const std::string ts_path = std::string(DATA_DIR) + "/vgg16b1.pt";
    torch::jit::Module module = torch::jit::load(ts_path);

    size_t paramBytes = 0;
    for (const auto& p : module.named_parameters(/*recurse=*/true)) {
      const auto& t = p.value;
      paramBytes += (size_t)t.numel() * t.element_size();
    }

    for (const auto& b : module.named_buffers(/*recurse=*/true)) {
      const auto& t = b.value;
      paramBytes += (size_t)t.numel() * t.element_size();
    }

    /*
      freeze to inline submodules & fold attributes
      If no freeze, it will create IR like %30 : Tensor = prim::CallMethod[name="forward"](%relu.1, %26) # :0:0
      which hides all the nodes (relu) behind
      This will save both frozend version and original JIT module, so memory occupation increase.
    */
    module = torch::jit::freeze_module(std::move(module));

    auto method = module.get_method("forward");
    std::shared_ptr<torch::jit::Graph> graph = method.graph();

    // Seed concrete input type (use CPU or CUDA as you prefer)
    // auto* xin = graph->inputs().at(1);
    // at::Tensor dummy = at::empty({1,3,224,224}, at::device(c10::kCPU).dtype(at::kFloat));
    // xin->setType(c10::TensorType::create(dummy));

    // std::cout << "Checking graph : " << graph->toString() << "\n";
    // std::cout << std::endl;

    /*
    1. Inline to flatten the graph so it looks like IR (Just like one straigh list of ops)

    2. Constant propagation : folds constatns through graph
    	- If a node’s outputs depend only on known constants, it precomputes and replaces with literal constants.=
	    - It prune dead constant subgraphs

    3. PropagateInputShapes : populate graph types values for each tensor value(like %10, %11)
      - Applies per-op rules (e.g., linear, relu, conv) to infer TensorType (dtype, device, sizes)
      - Writes the inferred TensorType onto each Value

      Which makes each TensorType has
        - Scalar Type(float, double)
        - sizes (8 or 16 or ...)
        - device info and flags like require grads

      Before :
        %10 : Tensor = aten::linear(%x.1, %weight.2, %bias.2)
        %11 : Tensor = aten::relu(%10)
        %14 : Tensor = aten::linear(%11, %weight.1, %bias.1)

      After : 
        %x.1  : Float(8, 32, ...)
        %10   : Float(8, 64, ...)
        %11   : Float(8, 64, ...)
        %14   : Float(8, 10, ...)
    */

    torch::jit::Inline(*graph);
    torch::jit::ConstantPropagation(graph);
    torch::jit::PropagateInputShapes(graph);

    /*
      Sum the parameters

      module.named_parameters (learnable, gradient optimizable)
        : weights and bias (doesn't free memory as pass)
        : return iterator (name, tensor) of every pairs for registerd parameters
        ex) fc1.weight [64, 32], fc1.bias [64], etc.

      module.named_buffers (non-learnable, no gradients)
        : batch norm calculating mean and variance, positional encoding, masking
        : Same logic as parameters, but it will return nothing for current simple implementation

      In detail

      For parameters, if written like 
        self.fc1 = nn.Linear(32, 64)

      Then, it will be
        self.fc1.weight = nn.Parameter(torch.randn(64, 32))
        self.fc1.bias = nn.Parameter(torch.randn(64))
    
      nn.Parameter is a subclass of Tensor,
      it gets auto-registered in model.parameters().

      But !!! buffers are save explicity like below.
      self.register_buffer("running_mean", torch.zeros(8))

      Still both of them saved like below.
      model._parameters  # dict of name → tensor
      model._buffers     # dict of name → tensor
    */

    // size_t paramBytes = 0;
    // for (const auto& p : module.named_parameters(/*recurse=*/true)) {
    //   const auto& t = p.value;
    //   paramBytes += static_cast<size_t>(t.numel()) * t.element_size();
    // }
    // for (const auto& b : module.named_buffers(/*recurse=*/true)) {
    //   const auto& t = b.value;
    //   paramBytes += static_cast<size_t>(t.numel()) * t.element_size();
    // }

    /*
      Topological ordering
      Since simple MLP is simple, the order is as it is shown in ir.txt

      Node (torch::jit::Node) is object inside Graph(torch::jit::Graph) which represent each ops
      (like aten::linear, aten::relu, prim::GetAttr, …)

      Graph stores these nodes in a topologically ordered linked list
      - Every node appear after all the nodes that produce its inputs, so we can safely
        execute them sequentially from top to bottom.

      Ex)
      %10 = aten::linear(%x.1, %weight.2, %bias.2)
      %11 = aten::relu(%10)
      %14 = aten::linear(%11, %weight.1, %bias.1)

      Inside works like below.
      - Node 1 (aten::linear) → produces %10
	    - Node 2 (aten::relu) → consumes %10, produces %11
	    - Node 3 (aten::linear) → consumes %11, produces %14
      - ...

      As topological ordering holds, we don't hit any undefined inputs.
    */

    /*
      we save them in vector later we need random access (die or live)
      (e.g. to find a node’s numeric position, or look up by index),
      which the graph->nodes() iterator doesn’t support.
    */

    std::vector<torch::jit::Node*> topo;
    std::unordered_map<torch::jit::Node*, size_t> nodeIdx;
    size_t idx = 0;
    for (auto* n : graph->nodes()) {
      topo.push_back(n);
      nodeIdx[n] = idx++;
    }

    /*
      Checking liveness : to track when is the last use for each tensor output (how long alive)
      topo : node in topological order
      nodeIdx : fast look up for A node*. at what step does node N run?

      last use : where tensor die??
    */
    auto lastUse = compute_last_use_indices(topo, nodeIdx);

    size_t liveBytes = 0;         // running total of all live active bytes
    size_t peakActBytes = 0;      // peak active liveBytes ever reaches

    struct LiveVal {
      const torch::jit::Value* v; // SSA
      size_t bytes;               // size in bytes
      size_t dieAt;               // where it dies
    };
    std::vector<LiveVal> liveSet;

    for (size_t i = 0; i < topo.size(); ++i) {
      auto* n = topo[i];

      // New outputs become live
      for (auto* v : n->outputs()) {
        auto tensorType = v->type()->cast<c10::TensorType>();
        if (!tensorType) continue; // skip non tensor

        auto tensorSt = tensorType->scalarType();
        if (!tensorSt.has_value()) continue; // skip unknown dtype

        size_t bytes = tensor_type_nbytes(tensorType->sizes(), *tensorSt);
        if (bytes == 0) continue; // skiip unknown shape

        size_t dieAt = lastUse.count(v) ? lastUse[v] : i; // default is die now (ex last node)
        liveSet.push_back({v, bytes, dieAt});
        liveBytes += bytes;

        if (liveBytes > peakActBytes)
          peakActBytes = liveBytes;
      }

      // Kill values whose last use is this node
      for (auto it = liveSet.begin(); it != liveSet.end();) {
        if (it->dieAt == i) {
          liveBytes -= it->bytes;
          it = liveSet.erase(it);

        } else {
          ++it;
        }
      }
    }

    const size_t totalPeak = paramBytes + peakActBytes;
    std::cout << "Parameter bytes:        " << pretty(paramBytes) << "\n";
    std::cout << "Peak activations:       " << pretty(peakActBytes) << "\n";
    std::cout << "Total peak (inference): " << pretty(totalPeak) << "\n";
    return 0;

  } catch (const c10::Error& e) {
    std::cerr << "Error: " << e.msg() << std::endl;
    return 1;
  }
}

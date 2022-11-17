/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_CONVERTLMHLOTOCPURUNTIMEPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::lmhlo::CustomCallOp;
using mlir::lmhlo::InfeedOp;
using mlir::lmhlo::OutfeedOp;

using xla_cpu::PartitionIdOp;
using xla_cpu::ReplicaIdOp;

using xla::runtime::AppendCustomCallAttrs;
using xla::runtime::CustomCallDeclarations;

class ConvertLmhloToCpuRuntimePass
    : public impl::ConvertLmhloToCpuRuntimePassBase<
          ConvertLmhloToCpuRuntimePass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }
};

//===----------------------------------------------------------------------===//

class CustomCallOpLowering : public OpRewritePattern<CustomCallOp> {
 private:
  static constexpr const char kCustomCallTarget[] = "xla.cpu.custom_call";

 public:
  CustomCallOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // By default all operands passed to the custom call handler.
    llvm::SmallVector<Value> operands = op.getOperands();

    // Get the number of outputs from operand_segment_sizes.
    int64_t num_results = op->getAttrOfType<DenseI32ArrayAttr>(
        op.getOperandSegmentSizesAttrName())[1];

    // If custom call has target arguments mapping, then we need to pass empty
    // memrefs in place of holes.
    if (op.getTargetArgMapping().has_value()) {
      auto mapping = *op.getTargetArgMapping();
      int64_t num_args = mapping.getNumArgs();
      num_results = mapping.getNumResults();

      // Always create an `alloca` in the parent function entry block.
      // See: https://llvm.org/docs/Frontend/PerformanceTips.html#use-of-allocas
      Value hole = [&]() -> Value {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(
            &op->getParentOfType<func::FuncOp>().front());
        return b.create<memref::AllocaOp>(MemRefType::get({0}, b.getI8Type()));
      }();

      // We represent holes as empty i8 memrefs.
      operands = llvm::SmallVector<Value>(num_args + num_results, hole);

      // Update operands to mapped custom call arguments.
      auto args = mapping.getArgsToTargetArgs();
      for (const auto& indexed : llvm::enumerate(args))
        operands[indexed.value()] = op.getArgs()[indexed.index()];

      // Update operands to mapped custom call results.
      auto res = mapping.getResultsToTargetResults();
      for (const auto& indexed : llvm::enumerate(res))
        operands[num_args + indexed.value()] = op.getOutput()[indexed.index()];
    }

    // Create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCustomCallTarget, TypeRange(ValueRange(operands)), TypeRange());

    llvm::SmallVector<NamedAttribute> custom_call_attrs = {
        {b.getStringAttr("num_results"),
         b.getI32IntegerAttr(static_cast<int32_t>(num_results))},
        {b.getStringAttr("api_version"), op.getApiVersionAttr()},
        {b.getStringAttr("call_target_name"), op.getCallTargetNameAttr()}};

    // Call the runtime intrinsic with the original operands.
    auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee.getName(), TypeRange(), operands);
    AppendCustomCallAttrs(call, custom_call_attrs);

    return success();
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

LogicalResult LowerXfeed(Operation* op, PatternRewriter& rewriter,
                         StringRef call_target,
                         CustomCallDeclarations& custom_calls) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  // By default all operands are passed to the custom call handler.
  llvm::SmallVector<Value> operands = op->getOperands();

  // Create a custom call function declaration.
  func::FuncOp callee = custom_calls.GetOrCreate(
      b, call_target, TypeRange(ValueRange(operands)), TypeRange());

  // Call the runtime intrinsic with the original operands.
  rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(), TypeRange(),
                                            operands);

  return success();
}

class InfeedOpLowering : public OpRewritePattern<InfeedOp> {
 private:
  static constexpr const char kCallTarget[] = "xla.cpu.infeed";

 public:
  InfeedOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(InfeedOp op,
                                PatternRewriter& rewriter) const override {
    return LowerXfeed(op, rewriter, kCallTarget, custom_calls_);
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class OutfeedOpLowering : public OpRewritePattern<OutfeedOp> {
 private:
  static constexpr const char kCallTarget[] = "xla.cpu.outfeed";

 public:
  OutfeedOpLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(OutfeedOp op,
                                PatternRewriter& rewriter) const override {
    return LowerXfeed(op, rewriter, kCallTarget, custom_calls_);
  }

 private:
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

template <typename IdOp>
class IdOpLowering : public OpRewritePattern<IdOp> {
 public:
  IdOpLowering(MLIRContext* ctx, llvm::StringRef call_target,
               CustomCallDeclarations& custom_calls)
      : OpRewritePattern<IdOp>(ctx),
        call_target_(call_target),
        custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(IdOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // Create a custom call function declaration.
    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, call_target_, TypeRange(), TypeRange(rewriter.getI32Type()));

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee.getName(),
                                              TypeRange(rewriter.getI32Type()));
    return success();
  }

 private:
  llvm::StringRef call_target_;
  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

class AllReduceLowering : public OpRewritePattern<xla_cpu::AllReduceOp> {
 public:
  AllReduceLowering(MLIRContext* ctx, CustomCallDeclarations& custom_calls)
      : OpRewritePattern(ctx), custom_calls_(custom_calls) {}

  LogicalResult matchAndRewrite(xla_cpu::AllReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getOperandTypes().front().isa<MemRefType>()) {
      return failure();
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    b.setInsertionPoint(op);

    // Subview ops result in strided Memrefs. The runtime can't deal with them,
    // so we copy everything that doesn't have the default layout.
    SmallVector<Value> new_operands;
    SmallVector<Type> new_operand_types;
    for (Value operand : op.getOperands()) {
      auto ty = operand.getType().cast<MemRefType>();
      if (ty.getLayout().isIdentity()) {
        new_operand_types.push_back(ty);
        new_operands.push_back(operand);
      } else {
        auto default_layout_ty =
            MemRefType::get(ty.getShape(), ty.getElementType());
        new_operand_types.push_back(default_layout_ty);
        auto new_operand = new_operands.emplace_back(
            b.create<memref::AllocOp>(default_layout_ty));
        b.create<memref::CopyOp>(operand, new_operand);
      }
    }

    func::FuncOp callee = custom_calls_.GetOrCreate(
        b, kCallTarget, new_operand_types, TypeRange());
    auto call =
        b.create<func::CallOp>(callee.getName(), TypeRange(), new_operands);

    // Set default attributes and copy attributes from original op.
    call->setAttr("use_global_device_ids", b.getI32IntegerAttr(0));
    call->setAttr("op_id", b.getI64IntegerAttr(0));
    for (auto& attr : op->getAttrs()) {
      call->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.eraseOp(op);
    return success();
  }

 private:
  static constexpr const char kCallTarget[] = "xla.cpu.all_reduce";

  CustomCallDeclarations& custom_calls_;
};

//===----------------------------------------------------------------------===//

void ConvertLmhloToCpuRuntimePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Keep track of the custom calls created from the lowered operations.
  SymbolTable sym_table(module);
  CustomCallDeclarations custom_calls(std::move(sym_table));

  // Convert lmhlo operations to XLA cpu runtime custom calls.
  RewritePatternSet patterns(ctx);
  patterns.insert<InfeedOpLowering, OutfeedOpLowering, CustomCallOpLowering,
                  AllReduceLowering>(ctx, custom_calls);
  patterns.insert<IdOpLowering<PartitionIdOp>>(ctx, "xla.cpu.partition_id",
                                               custom_calls);
  patterns.insert<IdOpLowering<ReplicaIdOp>>(ctx, "xla.cpu.replica_id",
                                             custom_calls);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLmhloToCpuRuntimePass() {
  return std::make_unique<ConvertLmhloToCpuRuntimePass>();
}

}  // namespace cpu
}  // namespace xla

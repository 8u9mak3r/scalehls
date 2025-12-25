//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "scalehls/Transforms/Passes.h"
#include "scalehls/Transforms/Utils.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

static void updateSubFuncs(func::FuncOp func, Builder builder) {
  func.walk([&](func::CallOp op) {
    auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
    auto subFunc = dyn_cast<func::FuncOp>(callee);

    // Set sub-function type.
    auto subResultTypes = op.getResultTypes();
    auto subInputTypes = op.getOperandTypes();
    auto newType = builder.getFunctionType(subInputTypes, subResultTypes);

    if (subFunc.getFunctionType() != newType) {
      subFunc.setType(newType);

      // Set arguments type.
      unsigned index = 0;
      for (auto inputType : op.getOperandTypes())
        subFunc.getArgument(index++).setType(inputType);

      // Set results type.
      auto returnOp = cast<func::ReturnOp>(subFunc.front().getTerminator());
      index = 0;
      for (auto resultType : op.getResultTypes())
        returnOp.getOperand(index++).setType(resultType);

      // Recursively apply array partition strategy.
      updateSubFuncs(subFunc, builder);
    }
  });
}

/// Apply the specified array partition factors and kinds.
bool scalehls::applyArrayPartition(Value array, ArrayRef<unsigned> factors,
                                   ArrayRef<hls::PartitionKind> kinds,
                                   bool updateFuncSignature) {
  auto builder = Builder(array.getContext());
  auto arrayType = array.getType().dyn_cast<MemRefType>();
  if (!arrayType || !arrayType.hasStaticShape() ||
      (int64_t)factors.size() != arrayType.getRank() ||
      (int64_t)kinds.size() != arrayType.getRank())
    return false;

  // Walk through each dimension of the current memory.
  SmallVector<AffineExpr, 4> partitionIndices;
  SmallVector<AffineExpr, 4> addressIndices;

  for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
    auto kind = kinds[dim];
    auto factor = factors[dim];

    if (kind == PartitionKind::CYCLIC) {
      partitionIndices.push_back(builder.getAffineDimExpr(dim) % factor);
      addressIndices.push_back(builder.getAffineDimExpr(dim).floorDiv(factor));

    } else if (kind == PartitionKind::BLOCK) {
      auto blockFactor = (arrayType.getShape()[dim] + factor - 1) / factor;
      partitionIndices.push_back(
          builder.getAffineDimExpr(dim).floorDiv(blockFactor));
      addressIndices.push_back(builder.getAffineDimExpr(dim) % blockFactor);

    } else {
      partitionIndices.push_back(builder.getAffineConstantExpr(0));
      addressIndices.push_back(builder.getAffineDimExpr(dim));
    }
  }

  // Construct new layout map.
  partitionIndices.append(addressIndices.begin(), addressIndices.end());
  auto layoutMap = AffineMap::get(arrayType.getRank(), 0, partitionIndices,
                                  builder.getContext());

  // Construct new array type.
  auto newType =
      MemRefType::get(arrayType.getShape(), arrayType.getElementType(),
                      layoutMap, arrayType.getMemorySpace());

  // Set new type.
  array.setType(newType);

  // if (updateFuncSignature)
  //   if (auto func =
  //           dyn_cast<func::FuncOp>(array.getParentBlock()->getParentOp())) {
  //     // Align function type with entry block argument types only if the array
  //     // is defined as an argument of the function.
  //     if (!array.getDefiningOp()) {
  //       auto resultTypes = func.front().getTerminator()->getOperandTypes();
  //       auto inputTypes = func.front().getArgumentTypes();
  //       func.setType(builder.getFunctionType(inputTypes, resultTypes));
  //     }

  //     // Update the types of all sub-functions.
  //     updateSubFuncs(func, builder);
  //   }
  return true;
}

static AffineMap getIdentityAffineMap(const SmallVectorImpl<Value> &operands,
                                      unsigned rank, MLIRContext *context) {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(rank);
  unsigned dimCount = 0;
  unsigned symbolCount = 0;

  for (auto operand : operands) {
    if (isValidDim(operand))
      exprs.push_back(getAffineDimExpr(dimCount++, context));
    else if (isValidSymbol(operand))
      exprs.push_back(getAffineSymbolExpr(symbolCount++, context));
    else
      return AffineMap();
  }
  return AffineMap::get(dimCount, symbolCount, exprs, context);
}

static AffineValueMap getAffineValueMap(Operation *op) {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  SmallVector<Value, 4> operands;
  if (auto loadOp = dyn_cast<mlir::AffineReadOpInterface>(op)) {
    operands = loadOp.getMapOperands();
    map = loadOp.getAffineMap();

  } else if (auto storeOp = dyn_cast<mlir::AffineWriteOpInterface>(op)) {
    operands = storeOp.getMapOperands();
    map = storeOp.getAffineMap();

  } else if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    operands = readOp.getIndices();
    map = getIdentityAffineMap(operands, readOp.getShapedType().getRank(),
                               readOp.getContext());
  } else {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    operands = writeOp.getIndices();
    map = getIdentityAffineMap(operands, writeOp.getShapedType().getRank(),
                               writeOp.getContext());
  }

  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  return AffineValueMap(map, operands);
}

static SmallVector<AffineMap, 4>
getDimAccessMaps(Operation *op, AffineValueMap valueMap, int64_t dim) {
  // Only keep the mapping result of the target dimension.
  auto baseMap = AffineMap::get(valueMap.getNumDims(), valueMap.getNumSymbols(),
                                valueMap.getResult(dim));

  // Get the permuation map from the transfer read/write op.
  AffineMap permuteMap;
  ArrayRef<int64_t> vectorShape;
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    permuteMap = readOp.getPermutationMap();
    vectorShape = readOp.getVectorType().getShape();
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    permuteMap = writeOp.getPermutationMap();
    vectorShape = writeOp.getVectorType().getShape();
  }

  SmallVector<AffineMap, 4> maps({baseMap});
  if (!permuteMap)
    return maps;

  // Traverse each dimension of the transfered vector.
  for (unsigned i = 0, e = permuteMap.getNumResults(); i < e; ++i) {
    auto dimExpr = permuteMap.getResult(i).dyn_cast<AffineDimExpr>();

    // If the permutation result of the current dimension is equal to the target
    // dimension, we push back the access map of each element of the vector into
    // the "maps" to be returned.
    if (dimExpr && dimExpr.getPosition() == dim) {
      for (int64_t offset = 0, size = vectorShape[i]; offset < size; ++offset) {
        auto map = AffineMap::get(baseMap.getNumDims(), baseMap.getNumSymbols(),
                                  baseMap.getResult(0) + offset);
        maps.push_back(map);
      }
      break;
    }
  }
  return maps;
}

namespace
{
using PartitionInfo = std::pair<PartitionKind, int64_t>;
/// Partition info are decided by the induction loop variable affiliated to the innermost loop layer;
std::pair<PartitionInfo, int64_t>
calculateDimAccessDistance(AffineExpr accessExpr, Operation* accessOp, DenseMap<AffineExpr, Value>& exprOperandMap) {
  auto partition = std::pair<PartitionInfo, int64_t>();

  if (auto binaryOpExpr = accessExpr.dyn_cast<AffineBinaryOpExpr>()) {
    auto lhs = binaryOpExpr.getLHS();
    auto rhs = binaryOpExpr.getRHS();
    auto exprKind = binaryOpExpr.getKind();
    auto partitionLHS = calculateDimAccessDistance(lhs, accessOp, exprOperandMap);
    auto partitionRHS = calculateDimAccessDistance(rhs, accessOp, exprOperandMap);

    // return the one decided by an inner loop
    int64_t levelLHS = partitionLHS.second;
    int64_t levelRHS = partitionRHS.second;
    partition = levelLHS > levelRHS ? partitionLHS : partitionRHS;

    if (partition.first.first == PartitionKind::NONE) {
      switch (exprKind) {
        // the most common cases
        case AffineExprKind::Add:
          partition.first.first = PartitionKind::CYCLIC;
          break;

        case AffineExprKind::Mul:
          partition.first.first = PartitionKind::BLOCK;
          break;
      
        default:
          break;
      }
    }
  } else if (auto dimExpr = accessExpr.dyn_cast<AffineDimExpr>()) {
    auto dimInductionVar = exprOperandMap[dimExpr];
    auto loop = dyn_cast<AffineForOp>(accessOp->getParentOp());
    int64_t level = 0;

    while (loop != nullptr) {
      auto loopInductionVar = loop.getInductionVar();
      if (loopInductionVar == dimInductionVar) {
        // calculate the level of this decisive loop
        auto outerLoop = dyn_cast<AffineForOp>(loop->getParentOp());
        while(outerLoop != nullptr) {
          level += 1;
          outerLoop = dyn_cast<AffineForOp>(outerLoop->getParentOp());
        }
        break;
      }

      loop = dyn_cast<AffineForOp>(loop->getParentOp());
    }
    // Partition kind will later be decided by the AffineExprKind of parent affine expression 
    partition = std::pair<PartitionInfo, int64_t>(PartitionInfo(PartitionKind::NONE, loop.getConstantUpperBound()), level);
  } else {
    assert(accessExpr.isa<AffineConstantExpr>() || accessExpr.isa<AffineSymbolExpr>());
    partition = std::pair<PartitionInfo, int64_t>(PartitionInfo(PartitionKind::NONE, 1), -1);
  }

  return partition;
}
} // namespace

namespace
{
using MemAccessOpWithAffineValueMap = std::pair<Operation*, AffineValueMap>;

/// Started from the top func, to collect load and store operations in every subfunc and return them in "map".
void getMemAccessOpsInfo(func::FuncOp& funcOp, SmallVector<MemAccessOpWithAffineValueMap, 8>& ops,
                                                                      unsigned operandIdx) {
  auto arg = funcOp.getArgument(operandIdx);
  // auto funcName = funcOp.getNameAttr();
  
  for (auto& use : arg.getUses()) {
    auto user = use.getOwner();

    if (isa<func::CallOp>(user)) {
      auto callOp = dyn_cast<func::CallOp>(user);
      auto callee = SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
      auto subFunc = dyn_cast<func::FuncOp>(callee);
      assert(subFunc && "callable is not a function operation");
      unsigned idx = use.getOperandNumber();
      getMemAccessOpsInfo(subFunc, ops, idx);
    } else if (isa<AffineLoadOp, AffineStoreOp>(user)) {
      auto affineValueMap = getAffineValueMap(user);
      ops.push_back(MemAccessOpWithAffineValueMap(user, affineValueMap));
    } else {
      llvm_unreachable("......\n");
    }
  }
}
} // namespace

/// Find the suitable array partition factors and kinds for the input buffer
static std::pair<SmallVector<PartitionKind>, SmallVector<unsigned>>
getArrayPartition (Value buffer) {
  SmallVector<MemAccessOpWithAffineValueMap, 8> memAccessOpsInfo;

  for (auto& use : buffer.getUses()) {
    auto user = use.getOwner();
    if (auto callOp = dyn_cast<func::CallOp>(user)) {
      auto callee = SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
      auto subFunc = dyn_cast<func::FuncOp>(callee);
      assert(subFunc && "callable is not a function operation");
      unsigned operandIdx = use.getOperandNumber();
      getMemAccessOpsInfo(subFunc, memAccessOpsInfo, operandIdx);
    }
  }

  llvm::dbgs() << "Buffer:\n";
  buffer.dump();
  SmallVector<PartitionKind> kinds;
  SmallVector<unsigned> factors;
  for (int64_t dim = 0; dim < buffer.getType().dyn_cast<MemRefType>().getRank(); dim += 1) {
    auto dimPartition = PartitionInfo(PartitionKind::NONE, 1);

    for (auto opInfo : memAccessOpsInfo) {
      auto accessOp = opInfo.first;
      auto accessAffineValueMap = opInfo.second;
      // For now we don't take vector ops into consideration, so the returned SmallVector only has one AffineValueMap in it.
      auto dimAccessMap = *getDimAccessMaps(accessOp, accessAffineValueMap, dim).begin();
      auto dimAccessAffinevalueMap = AffineValueMap(dimAccessMap, accessAffineValueMap.getOperands());
          
      SmallVector<AffineExpr> exprs;
      for (unsigned i = 0; i < dimAccessAffinevalueMap.getNumDims(); i += 1)
        exprs.push_back(mlir::getAffineDimExpr(i, buffer.getContext()));
      for (unsigned i = 0; i < dimAccessAffinevalueMap.getNumSymbols(); i += 1)
        exprs.push_back(mlir::getAffineSymbolExpr(i, buffer.getContext()));

      auto exprOperandMap = DenseMap<AffineExpr, Value>();
      for (auto expr : llvm::enumerate(exprs)) 
        exprOperandMap[expr.value()] = dimAccessAffinevalueMap.getOperand(expr.index());

      // llvm::dbgs() << "Dim access map:\n";
      // dimAccessAffinevalueMap.getAffineMap().dump();
      auto partition = calculateDimAccessDistance(dimAccessAffinevalueMap.getResult(0), accessOp, exprOperandMap).first;
      dimPartition = partition.second <= dimPartition.second ? dimPartition : partition;
      // llvm::dbgs() << "Partition Kind: " << (int)partition.first << " Partition factor: " << partition.second << "\n";
    }

      
    llvm::dbgs() << "Dim: " << dim << " Partition Kind: " << (int)dimPartition.first 
                    << " Partition factor: " << dimPartition.second << "\n";
    /// For now, we only apply one particular partition method to a certain dimension of the buffer, 
    /// and place corresponding HLS directive macros at its definition and every reference
    kinds.push_back(dimPartition.first);
    factors.push_back(dimPartition.second);
  }

  return {kinds, factors};
}


/// Find the suitable array partition factors and kinds for all arrays in the
/// targeted function.
bool scalehls::applyAutoArrayPartition(func::FuncOp func) {
  func.walk([&](hls::BufferOp buffer){
    auto partition = getArrayPartition(buffer);
    auto kinds = partition.first;
    auto factors = partition.second;
    applyArrayPartition(buffer, factors, kinds, /*updateFuncSignature=*/true);

    llvm::dbgs() << "\n\n\n";
  });

  for (auto funcArg : func.getArguments()) {
    auto partition = getArrayPartition(funcArg);
    auto kinds = partition.first;
    auto factors = partition.second;
    applyArrayPartition(funcArg, factors, kinds, /*updateFuncSignature=*/true);

    llvm::dbgs() << "\n\n\n";
  }

  Builder builder(func.getContext());
  auto resultTypes = func.front().getTerminator()->getOperandTypes();
  auto inputTypes = func.front().getArgumentTypes();
  func.setType(builder.getFunctionType(inputTypes, resultTypes));

  updateSubFuncs(func, builder);
  return true;
}

namespace {
struct ArrayPartition : public ArrayPartitionBase<ArrayPartition> {
  void runOnOperation() override {
    auto module = getOperation();

    // Get the top function.
    // FIXME: A better solution to handle the runtime main function.
    func::FuncOp topFunc;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (hasRuntimeAttr(func)) {
        topFunc = func;
        break;
      } else if (hasTopFuncAttr(func))
        topFunc = func;
    }

    if (!topFunc) {
      emitError(module.getLoc(), "fail to find the top function");
      return signalPassFailure();
    }
    applyAutoArrayPartition(topFunc);
  }
};
} // namespace

std::unique_ptr<Pass> scalehls::createArrayPartitionPass() {
  return std::make_unique<ArrayPartition>();
}

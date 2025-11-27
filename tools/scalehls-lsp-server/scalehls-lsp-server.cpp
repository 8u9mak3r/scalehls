#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "scalehls/InitAllDialects.h"
#include "scalehls/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::scalehls::registerAllDialects(registry);
  mlir::scalehls::registerAllPasses();

  return mlir::failed(mlir::MlirLspServerMain(
      argc, argv, registry));
}
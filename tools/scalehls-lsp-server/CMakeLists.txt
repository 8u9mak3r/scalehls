get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(scalehls-lsp-server
  scalehls-lsp-server.cpp
  )

llvm_update_compile_flags(scalehls-lsp-server)

target_link_libraries(scalehls-lsp-server
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRLspServerLib

  MLIRHLS
  MLIRScaleHLSTransforms

  # Threads::Threads
  )

function(initialize_submodule DIRECTORY)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/.git)
    find_package(Git QUIET REQUIRED)
    message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}/.git does not exist. Initializing ${DIRECTORY} submodule ...")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init ${DIRECTORY}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_EXIT_CODE)
    if(NOT GIT_EXIT_CODE EQUAL "0")
      message(FATAL_ERROR "${GIT_EXECUTABLE} submodule update --init dependencies/${DIRECTORY} failed with exit code ${GIT_EXIT_CODE}, please checkout submodules")
    endif()
  endif()
endfunction(initialize_submodule)


initialize_submodule(yaml-cpp)

# Apply patch to yaml-cpp to fix missing cstdint include
set(PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp-cstdint.patch")
set(TARGET_FILE "${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/src/emitterutils.cpp")

if(EXISTS ${PATCH_FILE} AND EXISTS ${TARGET_FILE})
  # Check if the patch has already been applied by looking for cstdint include
  file(READ ${TARGET_FILE} FILE_CONTENTS)
  string(FIND "${FILE_CONTENTS}" "#include <cstdint>" CSTDINT_FOUND)

  if(CSTDINT_FOUND EQUAL -1)
    message(STATUS "Applying yaml-cpp cstdint patch...")
    find_program(PATCH_PROGRAM patch)
    if(PATCH_PROGRAM)
      execute_process(
        COMMAND ${PATCH_PROGRAM} -p1 -i ${PATCH_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp
        RESULT_VARIABLE PATCH_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
      )
      if(PATCH_RESULT EQUAL 0)
        message(STATUS "yaml-cpp patch applied successfully")
      else()
        message(WARNING "Failed to apply yaml-cpp patch, trying manual fix...")
        # Fallback: manual insertion
        file(READ ${TARGET_FILE} ORIGINAL_CONTENT)
        string(REPLACE "#include <sstream>\n" "#include <sstream>\n#include <cstdint>\n" PATCHED_CONTENT "${ORIGINAL_CONTENT}")
        file(WRITE ${TARGET_FILE} "${PATCHED_CONTENT}")
        message(STATUS "Manual yaml-cpp fix applied")
      endif()
    else()
      message(STATUS "patch command not found, applying manual fix...")
      # Manual insertion if patch command is not available
      file(READ ${TARGET_FILE} ORIGINAL_CONTENT)
      string(REPLACE "#include <sstream>\n" "#include <sstream>\n#include <cstdint>\n" PATCHED_CONTENT "${ORIGINAL_CONTENT}")
      file(WRITE ${TARGET_FILE} "${PATCHED_CONTENT}")
      message(STATUS "Manual yaml-cpp fix applied")
    endif()
  else()
    message(STATUS "yaml-cpp cstdint patch already applied")
  endif()
endif()

add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)

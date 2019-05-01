# Copied from
# https://github.com/dev-cafe/cmake-cookbook/blob/master/chapter-07/recipe-03/c-cxx-example/set_compiler_flag.cmake
# Adapted from
# https://github.com/robertodr/ddPCM/blob/expose-C-api/cmake/custom/compilers/SetCompilerFlag.cmake
# which was adapted by Roberto Di Remigio from
# https://github.com/SethMMorton/cmake_fortran_template/blob/master/cmake/Modules/SetCompileFlag.cmake

# Given a list of flags, this stateless function will try each, one at a time,
# and set result to the first flag that works.
# If none of the flags works, result is "".
# If the REQUIRED key is given and no flag is found, a FATAL_ERROR is raised.
#
# Call is:
# set_compile_flag(result (Fortran|C|CXX) <REQUIRED> flag1 flag2 ...)
#
# Example:
# set_compiler_flag(working_compile_flag C REQUIRED "-Wall" "-warn all")

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckFortranCompilerFlag)

function(set_compiler_flag _result _lang)
  # build a list of flags from the arguments
  set(_list_of_flags)
  # also figure out whether the function
  # is required to find a flag
  set(_flag_is_required FALSE)
  foreach(_arg IN ITEMS ${ARGN})
    string(TOUPPER "${_arg}" _arg_uppercase)
    if(_arg_uppercase STREQUAL "REQUIRED")
      set(_flag_is_required TRUE)
    else()
      list(APPEND _list_of_flags "${_arg}")
    endif()
  endforeach()

  set(_flag_found FALSE)
  # loop over all flags, try to find the first which works
  foreach(flag IN ITEMS ${_list_of_flags})

    unset(_${flag}_works CACHE)
    if(_lang STREQUAL "C")
      check_c_compiler_flag("${flag}" _${flag}_works)
    elseif(_lang STREQUAL "CXX")
      check_cxx_compiler_flag("${flag}" _${flag}_works)
    elseif(_lang STREQUAL "Fortran")
      check_Fortran_compiler_flag("${flag}" _${flag}_works)
    else()
      message(FATAL_ERROR "Unknown language in set_compiler_flag: ${_lang}")
    endif()

    # if the flag works, use it, and exit
    # otherwise try next flag
    if(_${flag}_works)
      set(${_result} "${flag}" PARENT_SCOPE)
      set(_flag_found TRUE)
      break()
    endif()
  endforeach()

  # raise an error if no flag was found
  if(_flag_is_required AND NOT _flag_found)
    message(FATAL_ERROR "None of the required flags were supported")
  endif()
endfunction()

FILE(REMOVE_RECURSE
  "libcudbn.pdb"
  "libcudbn.so"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/cudbn.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)

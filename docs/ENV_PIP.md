

Install python packages with the following command. 

```

 source $(conda info --base)/etc/profile.d/conda.sh && conda activate hdvo && cat requirements_clean.txt | while read package; do    if [ -n "$package" ] && [[ ! "$package" =~ ^#.* ]]; then      echo "Installing: $package";     pip install "$package" || echo "Skip failed packages: $package";   fi; done

```

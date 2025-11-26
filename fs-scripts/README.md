# fairseq scripts

The shell scripts to train Custom NMT models using fairseq are included in this directory.
They are named in the format of either `en-xx.sh` or `xx-en.sh`, where `xx` is one of:

- `id` for Indonesian
- `ms` for Malay
- `ta` for Tamil
- `th` for Thai
- `tl` for Filipino
- `vi` for Vietnamese
- `zh` for Chinese

These scripts are broken down into multiple steps for preprocessing, training, and evaluation.
For example, in the `id-en.sh` script, the steps are:

1. Cloning moses and subword-nmt.
2. Preprocessing train and validation data.
3. Preprocessing test data.
4. (not in use)
5. Combining parallel texts in training data.
6. Learning BPE on the combined parallel texts.
7. Applying BPE to train, validation, and test data.
8. Clean train and validation data based on parallel text lengths and length ratio.
9. Copying test data to the same directory as train and validation data after Step 8.
10. Preprocessing train, validation, and test data using fairseq.
11. Training Custom NMT model using fairseq.
12. Generating using every model checkpoint on FLORES-101 devtest. This step is optional.
13. (not in use)
14. (not in use)
15. (not in use)
16. (not in use)
17. (not in use)
18. (not in use)
19. (not in use)
20. Computing evaluation scores for each checkpoint from the generated output of Step 12. This step is optional.

An example of running the script in sequence is shown in `/localhome/stanleyhan/mt/fs-scripts/example.sh`.
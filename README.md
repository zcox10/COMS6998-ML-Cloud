# COMS6998-ML-Cloud

Projects repo for the COMS6998 - ML Cloud class

For large file storage:

```bash
# In root
git lfs install 

# File types to store
git lfs track "*.zip"
git lfs track "*.bin"
git lfs track "some-large-model-file"

# Use git as normal when making commits
git add . && git commit -m "Sample commit message" && git push origin <branch_name>
```

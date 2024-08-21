git init
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track large files with Git LFS"
git add .
git commit -m "Commit for fraud detection project"
git checkout -b Rachit/Issue1
git remote add origin https://github.com/rachit-bhatt/Fraud-Detection
git push origin Rachit/Issue1
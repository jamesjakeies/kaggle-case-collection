# Publish To GitHub

如果你已经有 GitHub 仓库地址，在 `kaggle_cases` 目录执行：

```bash
git init
git add .
git commit -m "feat: add kaggle case collection"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

如果你还没有仓库，可以先在 GitHub 新建一个空仓库，再把仓库地址填到上面的 `<your-repo-url>`。

建议仓库名：

- `kaggle-case-collection`
- `kaggle-competition-starters`
- `kaggle-portfolio-projects`

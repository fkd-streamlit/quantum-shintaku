# GitHub公開手順

## 1. Gitリポジトリの初期化（まだの場合）

プロジェクトフォルダで以下のコマンドを実行：

```bash
git init
```

## 2. ファイルの追加

```bash
git add app.py
git add requirements.txt
git add README.md
git add .gitignore
git add 実行方法.md
git add GitHub公開手順.md
```

**注意**: Excelファイル（`*.xlsx`）は大きい可能性があるため、`.gitignore`で除外することを推奨します。
Excelファイルを含めたい場合は、明示的に追加：

```bash
git add quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify_with_lr022101.xlsx
```

## 3. 初回コミット

```bash
git commit -m "Initial commit: 量子神託 - 縁の球体（QUBO × アート）"
```

## 4. GitHubリポジトリの作成

1. GitHubにログイン
2. 右上の「+」ボタンから「New repository」を選択
3. リポジトリ名を入力（例：`quantum-shintaku`）
4. 説明を追加（例：「3D球体上でキーワードをQUBO最適化により配置するインタラクティブなWebアプリケーション」）
5. PublicまたはPrivateを選択
6. **「Initialize this repository with a README」はチェックしない**（既にREADME.mdがあるため）
7. 「Create repository」をクリック

## 5. リモートリポジトリの追加とプッシュ

GitHubで作成したリポジトリのURLを取得し、以下のコマンドを実行：

```bash
# リモートリポジトリを追加（YOUR_USERNAMEとYOUR_REPO_NAMEを置き換え）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# メインブランチを設定
git branch -M main

# プッシュ
git push -u origin main
```

## 6. 認証

GitHubへのプッシュ時に認証が求められる場合：

- **Personal Access Token (PAT)を使用する場合**:
  - GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
  - 新しいトークンを作成（`repo`スコープが必要）
  - パスワードの代わりにトークンを使用

- **SSHキーを使用する場合**:
  ```bash
  git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
  ```

## 7. 確認

GitHubのリポジトリページで、ファイルが正しくアップロードされているか確認してください。

## トラブルシューティング

### 既存のリポジトリに接続する場合

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### ファイルを更新した場合

```bash
git add .
git commit -m "Update: 変更内容の説明"
git push
```

### Excelファイルを除外したい場合

`.gitignore`に以下を追加（既に含まれています）：

```
*.xlsx
*.xls
```

その後、既に追跡されているファイルを削除：

```bash
git rm --cached quantum_shintaku_pack_v3_with_sense_20260213_oposite_modify_with_lr022101.xlsx
git commit -m "Remove Excel file from tracking"
git push
```

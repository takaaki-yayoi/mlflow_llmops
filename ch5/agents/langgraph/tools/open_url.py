"""システムブラウザでURLを開くツール。

このモジュールは、ユーザーのデフォルトブラウザで
URLを開くツールを提供します。
"""
import json
import subprocess
import sys

from langchain.tools import tool


@tool
def open_url(url: str) -> str:
    """システムのデフォルトブラウザでURLを開く。

    ドキュメントページ、GitHubリンク、その他のWeb URLを
    ユーザーが閲覧できるように開きます。

    Args:
        url: 開くURL

    Returns:
        結果を含むJSON文字列
    """
    try:
        # OSに応じたコマンドでブラウザを開く
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", url], check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", url], shell=True, check=True)
        else:  # Linux
            subprocess.run(["xdg-open", url], check=True)

        return json.dumps({"success": True, "message": f"URLを開きました: {url}"})

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

"""MLflow サンプルエージェントの CLI インターフェース。

このモジュールは、エージェントと対話するためのコマンドラインインターフェースを提供します。
ユーザーはテキスト入力でエージェントに質問し、ストリーミング形式で回答を受け取ることができます。

使用例:
    CLIを起動する::

        $ python -m cli.main

    または Makefile を使用::

        $ make cli

    対話例::

        > あなた: MLflowとは何ですか？
        > アシスタント: MLflowは、機械学習のライフサイクルを管理するための...

    利用可能なコマンド:
        /quit, /exit  - CLIを終了する
        /new          - 新しい会話スレッドを開始する
        /feedback     - 直前の回答にフィードバックを送る
"""
import sys
import io
import time
from pathlib import Path

import dotenv

from agents.langgraph.agent import LangGraphAgent
from agents.thread import Thread

# ターミナル出力用のANSIカラーコード
GREEN = "\033[92m"   # 緑色（推論過程の表示用）
AQUA = "\033[96m"    # 水色（アシスタントの回答用）
YELLOW = "\033[93m"  # 黄色（警告・コマンド表示用）
RED = "\033[91m"     # 赤色（エラー表示用）
RESET = "\033[0m"    # 色のリセット


def load_agent():
    """エージェントを初期化する。"""
    return LangGraphAgent()

def stream_text(text: str, delay: float = 0.005, color: str = RESET):
    """テキストをストリーミング効果で表示する。

    Args:
        text: 表示するテキスト
        delay: 各文字間の遅延時間（秒）。デフォルトは0.005秒。
        color: 表示色のANSIコード。デフォルトは白色。
    """
    sys.stdout.write(color)
    # 1文字ずつ出力してタイプライター効果を演出
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(RESET + "\n")
    sys.stdout.flush()


class GreenOutput(io.TextIOBase):
    """標準出力を緑色でラップするカスタム出力クラス。

    エージェントの推論過程を緑色で表示するために使用します。
    sys.stdoutを一時的にこのクラスのインスタンスに置き換えることで、
    エージェント内部のprint文も緑色で表示されます。
    """
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, text):
        # 空白以外のテキストのみ緑色で表示
        if text and text.strip():
            self.original_stdout.write(f"{GREEN}{text}{RESET}")
        else:
            self.original_stdout.write(text)
        return len(text)

    def flush(self):
        self.original_stdout.flush()


def print_banner():
    """起動時のバナーを表示する。"""
    print()
    print(f"{AQUA}╔══════════════════════════════════════════╗{RESET}")
    print(f"{AQUA}║       *    .  *       .   *              ║{RESET}")
    print(f"{AQUA}║ .  * {RESET}MLflow エージェントへようこそ！{AQUA}*  . ║{RESET}")
    print(f"{AQUA}║      *    .       *   .      *           ║{RESET}")
    print(f"{AQUA}╚══════════════════════════════════════════╝{RESET}")
    print()


def main():
    """CLIのメインエントリーポイント。

    環境変数の読み込み、エージェントの初期化、ユーザー入力のループ処理を行います。
    """
    # .envファイルの存在確認（APIキーなどの設定が必要）
    if not Path(".env").exists():
        print(f"\n{RED}エラー: .env ファイルが見つかりません{RESET}")
        print(f"\nAPIキーを含む .env ファイルを作成してください。")
        print(f".env.templateテンプレートを次のコマンドでコピーして、APIキーを保存してください: {YELLOW}cp .env.template .env{RESET}")
        print()
        sys.exit(1)

    # 環境変数を.envファイルから読み込み
    dotenv.load_dotenv()

    print_banner()

    # LangGraphエージェントのインスタンスを作成
    agent = load_agent()

    print()
    print(f"{YELLOW}コマンド:{RESET} /quit, /new, /feedback")
    print()

    # 会話スレッドを作成（会話履歴の管理用）
    thread = Thread()
    ctrl_c_count = 0  # Ctrl+C の連続押下回数をカウント。2回連続で押されたらセッションを終了する。
    last_response = None  # 直前の回答を保存（フィードバック用）

    # メインの対話ループ
    while True:
        original_stdout = sys.stdout
        try:
            # ユーザー入力を受け取る
            user_input = input("\nあなた: ").strip()
            ctrl_c_count = 0  # 入力があったらCtrl+Cカウントをリセット

            # 空入力はスキップ
            if not user_input:
                continue

            # 終了コマンドの処理
            if user_input in ("/quit", "/exit"):
                print("さようなら！")
                break

            # 新規スレッド開始コマンドの処理
            if user_input == "/new":
                thread = Thread()
                last_response = None
                print("\033[2J\033[H")  # ANSIエスケープで画面をクリア
                print(f"{YELLOW}新しいスレッドを開始しました{RESET}")
                continue

            # フィードバックコマンドの処理
            if user_input == "/feedback":
                if last_response is None:
                    print(f"{YELLOW}フィードバックする回答がありません{RESET}")
                    continue
                feedback = input("この回答は役に立ちましたか？ (y/n): ").strip().lower()
                if feedback in ("y", "yes"):
                    print(f"{GREEN}ポジティブなフィードバックをありがとうございます！{RESET}")
                elif feedback in ("n", "no"):
                    reason = input("改善点があれば教えてください（任意）: ").strip()
                    print(f"{GREEN}フィードバックをありがとうございます！{RESET}")
                else:
                    print(f"{YELLOW}フィードバックがキャンセルされました{RESET}")
                continue

            print(f"\n{GREEN}--- 推論中 ---{RESET}\n")

            # 標準出力を緑色出力に切り替えて推論過程を表示
            sys.stdout = GreenOutput(original_stdout)
            # エージェントにクエリを送信して回答を取得
            response = agent.process_query(user_input, thread)
            # 標準出力を元に戻す
            sys.stdout = original_stdout
            last_response = response

            print(f"\n{GREEN}---{RESET}\n")
            sys.stdout.write(f"{AQUA}アシスタント:{RESET} ")
            sys.stdout.flush()
            # 回答をストリーミング表示
            stream_text(response, color=AQUA)

        except KeyboardInterrupt:
            # Ctrl+C が押された場合の処理
            sys.stdout = original_stdout
            sys.stdout.write(RESET + "\n")
            sys.stdout.flush()

            ctrl_c_count += 1
            # 2回連続でCtrl+Cが押されたら終了
            if ctrl_c_count >= 2:
                print("\nさようなら！")
                break
            print(f"\n{YELLOW}（もう一度 Ctrl+C を押すと終了します）{RESET}")
            continue

        except Exception as e:
            # その他のエラーをキャッチして表示
            sys.stdout = original_stdout
            sys.stdout.write(RESET)
            sys.stdout.flush()
            print(f"\nエラー: {e}")


if __name__ == "__main__":
    main()

"""
第8章 8.1.3: グレースフルシャットダウン

非同期ログ記録を使用する場合、アプリケーション終了時に
バッファ内のトレースをフラッシュする必要があります。
"""

import atexit
import signal
import sys
import mlflow


class GracefulShutdownHandler:
    """
    アプリケーション終了時にトレースをフラッシュするハンドラー
    """
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Args:
            timeout_seconds: フラッシュ完了を待つ最大秒数
        """
        self.timeout_seconds = timeout_seconds
        self._registered = False
    
    def register(self) -> None:
        """
        シグナルハンドラーとatexit関数を登録します。
        """
        if self._registered:
            return
        
        # シグナルハンドラの登録
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # atexit登録 (正常終了時)
        atexit.register(self._graceful_shutdown)
        
        self._registered = True
        print("Graceful shutdown handler registered")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        シグナル受信時のハンドラー
        """
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}, initiating graceful shutdown...")
        self._graceful_shutdown()
        sys.exit(0)
    
    def _graceful_shutdown(self) -> None:
        """
        トレースバッファをフラッシュしてクリーンに終了
        """
        print("Flushing pending traces...")
        try:
            mlflow.flush_trace_async_logging(terminate=True)
            print("Trace flushing complete.")
        except Exception as e:
            print(f"Warning: Error during trace flush: {e}")


# シンプルな関数版
def setup_graceful_shutdown() -> None:
    """
    シンプルなグレースフルシャットダウン設定
    """
    def graceful_shutdown(signum=None, frame=None):
        print("Flushing pending traces...")
        mlflow.flush_trace_async_logging(terminate=True)
        print("Trace flushing complete.")
        if signum is not None:
            sys.exit(0)
    
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)


# 使用例
if __name__ == "__main__":
    import os
    import time
    
    # 非同期ログ記録を有効化
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
    
    # グレースフルシャットダウンを設定
    handler = GracefulShutdownHandler()
    handler.register()
    
    # または簡易版
    # setup_graceful_shutdown()
    
    print("Application running. Press Ctrl+C to test graceful shutdown.")
    
    # アプリケーションのメインループ (例)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

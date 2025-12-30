"""
第8章 8.4.5: ロールバック戦略

MLflow Model Registryのエイリアスを使用したロールバック実装です。

注意: MLflow 2.9以降、ステージ(Production/Staging/Archived)は非推奨です。
エイリアス(@champion, @challenger等)を使用してください。
"""

from datetime import datetime
from typing import Optional
from mlflow.tracking import MlflowClient


class ModelRollbackManager:
    """
    モデルのロールバックを管理するクラス
    
    使用例:
        manager = ModelRollbackManager("customer-support-agent")
        
        # ロールバック前の確認
        manager.get_alias_info("champion")
        
        # ロールバック実行
        manager.rollback(target_version="3", reason="品質劣化が検出されたため")
        
        # ロールバックの取り消し
        manager.undo_rollback()
    """
    
    def __init__(self, model_name: str, client: MlflowClient = None):
        """
        Args:
            model_name: 登録済みモデル名
            client: MLflowClient (指定しない場合は新規作成)
        """
        self.model_name = model_name
        self.client = client or MlflowClient()
        self._rollback_history: list[dict] = []
    
    def get_model_versions(self) -> list:
        """モデルの全バージョンを取得"""
        return self.client.search_model_versions(f"name='{self.model_name}'")
    
    def get_alias_info(self, alias: str) -> Optional[dict]:
        """
        エイリアスの情報を取得
        
        Args:
            alias: エイリアス名 (例: "champion", "challenger")
        
        Returns:
            エイリアス情報の辞書、またはNone
        """
        try:
            mv = self.client.get_model_version_by_alias(self.model_name, alias)
            return {
                "model_name": self.model_name,
                "alias": alias,
                "version": mv.version,
                "creation_timestamp": mv.creation_timestamp,
                "current_stage": getattr(mv, "current_stage", "N/A"),
                "run_id": mv.run_id,
                "tags": mv.tags,
            }
        except Exception as e:
            print(f"Alias '{alias}' not found: {e}")
            return None
    
    def list_aliases(self) -> dict[str, str]:
        """全エイリアスとそのバージョンを取得"""
        model = self.client.get_registered_model(self.model_name)
        aliases = {}
        
        for alias in getattr(model, "aliases", []):
            try:
                mv = self.client.get_model_version_by_alias(self.model_name, alias)
                aliases[alias] = mv.version
            except Exception:
                pass
        
        return aliases
    
    def rollback(
        self,
        target_version: str,
        alias: str = "champion",
        reason: str = None,
    ) -> dict:
        """
        モデルをロールバック
        
        Args:
            target_version: ロールバック先のバージョン
            alias: 更新するエイリアス (デフォルト: champion)
            reason: ロールバック理由
        
        Returns:
            ロールバック結果
        """
        # 現在のエイリアス情報を保存
        current_info = self.get_alias_info(alias)
        current_version = current_info["version"] if current_info else None
        
        # エイリアスを新しいバージョンに設定
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias=alias,
            version=target_version,
        )
        
        # ロールバック記録のタグを設定
        timestamp = datetime.now().isoformat()
        
        self.client.set_model_version_tag(
            name=self.model_name,
            version=target_version,
            key="rollback_from",
            value=str(current_version) if current_version else "unknown",
        )
        self.client.set_model_version_tag(
            name=self.model_name,
            version=target_version,
            key="rollback_timestamp",
            value=timestamp,
        )
        if reason:
            self.client.set_model_version_tag(
                name=self.model_name,
                version=target_version,
                key="rollback_reason",
                value=reason,
            )
        
        # 問題のあったバージョンに警告タグを設定
        if current_version:
            self.client.set_model_version_tag(
                name=self.model_name,
                version=current_version,
                key="status",
                value="rolled_back",
            )
            self.client.set_model_version_tag(
                name=self.model_name,
                version=current_version,
                key="rolled_back_at",
                value=timestamp,
            )
        
        # 履歴に記録
        rollback_record = {
            "timestamp": timestamp,
            "alias": alias,
            "from_version": current_version,
            "to_version": target_version,
            "reason": reason,
        }
        self._rollback_history.append(rollback_record)
        
        print(f"Rollback complete:")
        print(f"  Model: {self.model_name}")
        print(f"  Alias: @{alias}")
        print(f"  {current_version} → {target_version}")
        if reason:
            print(f"  Reason: {reason}")
        
        return rollback_record
    
    def undo_rollback(self, alias: str = "champion") -> Optional[dict]:
        """
        直前のロールバックを取り消し
        
        Args:
            alias: 対象のエイリアス
        
        Returns:
            取り消し結果、または履歴がない場合はNone
        """
        # 該当エイリアスの最新ロールバックを検索
        for record in reversed(self._rollback_history):
            if record["alias"] == alias and record["from_version"]:
                # 元のバージョンに戻す
                return self.rollback(
                    target_version=record["from_version"],
                    alias=alias,
                    reason=f"Undo rollback from {record['to_version']}",
                )
        
        print(f"No rollback history found for alias '@{alias}'")
        return None
    
    def get_rollback_history(self) -> list[dict]:
        """ロールバック履歴を取得"""
        return self._rollback_history.copy()


def quick_rollback(
    model_name: str,
    target_version: str,
    alias: str = "champion",
    reason: str = None,
) -> dict:
    """
    クイックロールバック関数
    
    Args:
        model_name: モデル名
        target_version: ロールバック先のバージョン
        alias: エイリアス
        reason: 理由
    
    Returns:
        ロールバック結果
    """
    manager = ModelRollbackManager(model_name)
    return manager.rollback(target_version, alias, reason)


def get_champion_version(model_name: str) -> Optional[str]:
    """現在のchampionバージョンを取得"""
    manager = ModelRollbackManager(model_name)
    info = manager.get_alias_info("champion")
    return info["version"] if info else None


# 使用例
if __name__ == "__main__":
    # 使用例のデモ
    model_name = "customer-support-agent"
    
    print("=== Model Rollback Example ===")
    print(f"Model: {model_name}")
    
    # 注意: 実際のモデルが必要
    # manager = ModelRollbackManager(model_name)
    
    # 現在の状態を確認
    # print("\nCurrent aliases:")
    # for alias, version in manager.list_aliases().items():
    #     print(f"  @{alias} → v{version}")
    
    # champion情報を確認
    # champion_info = manager.get_alias_info("champion")
    # if champion_info:
    #     print(f"\nChampion version: {champion_info['version']}")
    
    # ロールバック実行
    # manager.rollback(
    #     target_version="3",
    #     reason="品質スコアの低下が検出されたため"
    # )
    
    print("\nRollback example code ready. Uncomment to run with actual model.")

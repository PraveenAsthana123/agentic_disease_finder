"""
Utility Functions for Scheduled Sync, Export, and Backup
"""

import os
import json
import shutil
import schedule
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import zipfile


class ScheduledSync:
    """Scheduled automatic synchronization"""

    def __init__(self, rag_engine, sync_callback):
        """
        Initialize scheduled sync

        Args:
            rag_engine: RAG engine instance
            sync_callback: Function to call for sync
        """
        self.rag_engine = rag_engine
        self.sync_callback = sync_callback
        self.running = False
        self.thread = None
        self.schedule_config = None
        self.last_sync = None
        self.sync_history = []

    def schedule_daily(self, time_str: str = "02:00"):
        """
        Schedule daily sync

        Args:
            time_str: Time in HH:MM format
        """
        schedule.clear()
        schedule.every().day.at(time_str).do(self._run_sync)
        self.schedule_config = {
            'type': 'daily',
            'time': time_str,
            'enabled': True
        }
        self._start_scheduler()

    def schedule_weekly(self, day: str = "monday", time_str: str = "02:00"):
        """
        Schedule weekly sync

        Args:
            day: Day of week
            time_str: Time in HH:MM format
        """
        schedule.clear()
        getattr(schedule.every(), day).at(time_str).do(self._run_sync)
        self.schedule_config = {
            'type': 'weekly',
            'day': day,
            'time': time_str,
            'enabled': True
        }
        self._start_scheduler()

    def schedule_interval(self, hours: int = 6):
        """
        Schedule sync at intervals

        Args:
            hours: Hours between syncs
        """
        schedule.clear()
        schedule.every(hours).hours.do(self._run_sync)
        self.schedule_config = {
            'type': 'interval',
            'hours': hours,
            'enabled': True
        }
        self._start_scheduler()

    def _run_sync(self):
        """Execute sync operation"""
        print(f"[{datetime.now()}] Running scheduled sync...")

        try:
            result = self.sync_callback()

            sync_record = {
                'timestamp': datetime.now().isoformat(),
                'success': result.get('success', False),
                'synced_count': result.get('synced_count', 0),
                'errors': result.get('errors', [])
            }

            self.sync_history.append(sync_record)
            self.last_sync = datetime.now()

            print(f"Sync completed: {sync_record['synced_count']} papers synced")

        except Exception as e:
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
            self.sync_history.append(error_record)
            print(f"Sync failed: {e}")

    def _start_scheduler(self):
        """Start scheduler thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.thread.start()

    def _scheduler_loop(self):
        """Scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def stop(self):
        """Stop scheduler"""
        self.running = False
        schedule.clear()
        if self.thread:
            self.thread.join(timeout=5)

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'running': self.running,
            'config': self.schedule_config,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'next_run': str(schedule.next_run()) if schedule.jobs else None,
            'sync_history': self.sync_history[-10:]  # Last 10 syncs
        }


class ConversationExporter:
    """Export conversations to various formats"""

    @staticmethod
    def export_to_json(conversations: List[Dict], filepath: str):
        """
        Export to JSON

        Args:
            conversations: List of conversation exchanges
            filepath: Output file path
        """
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'conversations': conversations
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return filepath

    @staticmethod
    def export_to_markdown(conversations: List[Dict], filepath: str):
        """
        Export to Markdown

        Args:
            conversations: List of conversation exchanges
            filepath: Output file path
        """
        md_content = f"""# Conversation Export

**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Total Conversations:** {len(conversations)}

---

"""

        for i, conv in enumerate(conversations, 1):
            timestamp = conv.get('timestamp', 'Unknown')
            model = conv.get('model', 'Unknown')

            md_content += f"""## Conversation {i}

**Time:** {timestamp}
**Model:** {model}

### User Query
{conv.get('query', 'N/A')}

### Assistant Response
{conv.get('response', 'N/A')}

---

"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return filepath

    @staticmethod
    def export_to_html(conversations: List[Dict], filepath: str):
        """
        Export to HTML

        Args:
            conversations: List of conversation exchanges
            filepath: Output file path
        """
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Conversation Export</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .conversation {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .query {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        .response {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        h1 {{ color: #333; }}
        h3 {{ color: #555; margin-top: 0; }}
    </style>
</head>
<body>
    <h1>Conversation Export</h1>
    <p><strong>Exported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Total Conversations:</strong> {len(conversations)}</p>
    <hr>
"""

        for i, conv in enumerate(conversations, 1):
            timestamp = conv.get('timestamp', 'Unknown')
            model = conv.get('model', 'Unknown')

            html_content += f"""
    <div class="conversation">
        <div class="meta">
            <strong>Conversation {i}</strong> |
            Time: {timestamp} |
            Model: {model}
        </div>
        <div class="query">
            <h3>User Query</h3>
            <p>{conv.get('query', 'N/A')}</p>
        </div>
        <div class="response">
            <h3>Assistant Response</h3>
            <p>{conv.get('response', 'N/A').replace(chr(10), '<br>')}</p>
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filepath


class DatabaseBackup:
    """Backup and restore ChromaDB"""

    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize database backup

        Args:
            db_path: Path to ChromaDB directory
        """
        self.db_path = db_path
        self.backup_dir = "./backups"
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self, backup_name: Optional[str] = None) -> Dict:
        """
        Create database backup

        Args:
            backup_name: Optional backup name

        Returns:
            Backup information
        """
        if not os.path.exists(self.db_path):
            return {
                'success': False,
                'error': 'Database path does not exist'
            }

        # Generate backup name
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = os.path.join(self.backup_dir, backup_name)

        try:
            # Create zip archive
            zip_path = f"{backup_path}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.db_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.db_path)
                        zipf.write(file_path, arcname)

            # Get backup size
            backup_size = os.path.getsize(zip_path)

            # Save metadata
            metadata = {
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'size_bytes': backup_size,
                'size_mb': round(backup_size / (1024 * 1024), 2)
            }

            metadata_path = f"{backup_path}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'success': True,
                'backup_name': backup_name,
                'backup_path': zip_path,
                'size_mb': metadata['size_mb']
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def restore_backup(self, backup_name: str) -> Dict:
        """
        Restore database from backup

        Args:
            backup_name: Name of backup to restore

        Returns:
            Restore result
        """
        zip_path = os.path.join(self.backup_dir, f"{backup_name}.zip")

        if not os.path.exists(zip_path):
            return {
                'success': False,
                'error': f'Backup not found: {backup_name}'
            }

        try:
            # Backup current database
            current_backup = self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Clear current database
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)

            # Extract backup
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(self.db_path)

            return {
                'success': True,
                'backup_name': backup_name,
                'pre_restore_backup': current_backup
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def list_backups(self) -> List[Dict]:
        """
        List all available backups

        Returns:
            List of backup information
        """
        backups = []

        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.json'):
                metadata_path = os.path.join(self.backup_dir, filename)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        backups.append(metadata)
                except:
                    continue

        return sorted(backups, key=lambda x: x['created_at'], reverse=True)

    def delete_backup(self, backup_name: str) -> Dict:
        """
        Delete a backup

        Args:
            backup_name: Name of backup to delete

        Returns:
            Deletion result
        """
        zip_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
        json_path = os.path.join(self.backup_dir, f"{backup_name}.json")

        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(json_path):
                os.remove(json_path)

            return {
                'success': True,
                'backup_name': backup_name
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def auto_backup(self, max_backups: int = 7):
        """
        Create automatic backup and clean old backups

        Args:
            max_backups: Maximum number of backups to keep

        Returns:
            Backup result
        """
        # Create new backup
        result = self.create_backup()

        if result['success']:
            # Clean old backups
            backups = self.list_backups()

            if len(backups) > max_backups:
                # Delete oldest backups
                for old_backup in backups[max_backups:]:
                    self.delete_backup(old_backup['backup_name'])

        return result

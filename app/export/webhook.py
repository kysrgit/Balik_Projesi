"""
Webhook Bildirim Sistemi
Tespit olduğunda harici servislere HTTP POST bildirim gönderir.
Desteklenen hedefler: Slack, Discord, Teams, özel API endpoint'leri.
"""
import json
import time
import threading
from typing import Optional, Dict, List, Any
from urllib.request import Request, urlopen
from urllib.error import URLError


class WebhookNotifier:
    """
    Thread-safe webhook bildirim yöneticisi.
    
    Kullanım:
        notifier = WebhookNotifier()
        notifier.add_target("slack", "https://hooks.slack.com/...")
        notifier.notify(species="Lagocephalus sceleratus", confidence=0.85, lat=36.88, lon=30.70)
    """

    def __init__(self, rate_limit_seconds: float = 60.0):
        self._targets: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._last_notify_time: float = 0.0
        self._rate_limit = rate_limit_seconds
        self._stats = {'sent': 0, 'failed': 0, 'rate_limited': 0}

    def add_target(self, name: str, url: str) -> None:
        """Webhook hedefi ekle"""
        with self._lock:
            self._targets[name] = url

    def remove_target(self, name: str) -> None:
        """Webhook hedefi kaldır"""
        with self._lock:
            self._targets.pop(name, None)

    def get_targets(self) -> Dict[str, str]:
        """Mevcut hedefleri döndür"""
        with self._lock:
            return dict(self._targets)

    def get_stats(self) -> Dict[str, int]:
        """İstatistikleri döndür"""
        return dict(self._stats)

    def _format_payload(self, name: str, data: Dict[str, Any]) -> str:
        """Hedef tipine göre payload formatla"""
        species = data.get('species', 'Lagocephalus sceleratus')
        conf = data.get('confidence', 0)
        lat = data.get('lat', 'N/A')
        lon = data.get('lon', 'N/A')
        ts = data.get('timestamp', '')

        message = (
            f"🐡 Balon Balığı Tespiti!\n"
            f"Tür: {species}\n"
            f"Güven: {conf:.0%}\n"
            f"Konum: {lat}, {lon}\n"
            f"Zaman: {ts}"
        )

        url = self._targets.get(name, '')

        # Slack format
        if 'slack' in name.lower() or 'hooks.slack.com' in url:
            return json.dumps({"text": message})

        # Discord format
        if 'discord' in name.lower() or 'discord.com/api/webhooks' in url:
            return json.dumps({"content": message})

        # Teams format
        if 'teams' in name.lower() or 'webhook.office.com' in url:
            return json.dumps({
                "@type": "MessageCard",
                "summary": "Pufferfish Detection",
                "sections": [{
                    "activityTitle": "🐡 Balon Balığı Tespiti",
                    "facts": [
                        {"name": "Tür", "value": species},
                        {"name": "Güven", "value": f"{conf:.0%}"},
                        {"name": "Konum", "value": f"{lat}, {lon}"},
                        {"name": "Zaman", "value": ts}
                    ]
                }]
            })

        # Generic JSON
        return json.dumps({
            "event": "pufferfish_detection",
            "data": data,
            "source": "antigravity_detector"
        })

    def _send_one(self, name: str, url: str, payload: str) -> bool:
        """Tek bir hedefe gönder"""
        try:
            req = Request(
                url,
                data=payload.encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Antigravity-PufferfishDetector/1.0'
                },
                method='POST'
            )
            with urlopen(req, timeout=10) as resp:
                if resp.status < 300:
                    self._stats['sent'] += 1
                    return True
                else:
                    self._stats['failed'] += 1
                    return False
        except (URLError, Exception) as e:
            print(f"Webhook hata [{name}]: {e}")
            self._stats['failed'] += 1
            return False

    def notify(self, **kwargs) -> Dict[str, bool]:
        """
        Tüm kayıtlı hedeflere bildirim gönder.
        Rate limiting uygulanır.
        
        Args:
            species: Tür adı
            confidence: Güven skoru (0-1)
            lat: Enlem
            lon: Boylam
            timestamp: Zaman damgası
        
        Returns:
            {hedef_adı: başarılı_mı} sözlüğü
        """
        now = time.time()
        if now - self._last_notify_time < self._rate_limit:
            self._stats['rate_limited'] += 1
            return {}

        with self._lock:
            targets = dict(self._targets)

        if not targets:
            return {}

        self._last_notify_time = now
        results = {}

        data = {
            'species': kwargs.get('species', 'Lagocephalus sceleratus'),
            'confidence': kwargs.get('confidence', 0),
            'lat': kwargs.get('lat', None),
            'lon': kwargs.get('lon', None),
            'timestamp': kwargs.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
        }

        # Paralel gönderim
        threads = []
        for name, url in targets.items():
            payload = self._format_payload(name, data)
            t = threading.Thread(target=lambda n=name, u=url, p=payload: results.update({n: self._send_one(n, u, p)}))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15)

        return results

    def notify_async(self, **kwargs) -> None:
        """Arka planda bildirim gönder (ana thread'i bloklamaz)"""
        threading.Thread(target=self.notify, kwargs=kwargs, daemon=True).start()

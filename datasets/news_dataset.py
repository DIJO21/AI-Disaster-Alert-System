import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, List, Any
import requests
import json
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

class NewsDataset(Dataset):
    """
    Multilingual News feed parsing dataset for GDELT, ReliefWeb, and CrisisNLP
    with HuggingFace text tokenization.
    """
    def __init__(
        self,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
        size: int = 200,
        fetch_online: bool = False
    ):
        self.max_length = max_length
        self.size = size
        self.fetch_online = fetch_online
        
        if HAS_TOKENIZERS:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = MockTokenizer()
            
        # Parse data
        self.data = self._populate_feed()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            input_ids: Tokenized ID sequence
            attention_mask: Mask to filter out padding
            label: integer class representing disaster category
        """
        item = self.data[idx]
        text = item["text"]
        label = item["label"]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        return input_ids, attention_mask, label

    def _populate_feed(self) -> List[Dict[str, Any]]:
        feed = []
        if self.fetch_online:
            feed = self._fetch_reliefweb_reports()
            
        # If online fetch is empty, fallback to rich multilingual synthetic database
        if not feed:
            feed = self._generate_multilingual_synthetics()
            
        # Ensure exact size matching
        if len(feed) < self.size:
            multiplier = (self.size // len(feed)) + 1
            feed = (feed * multiplier)[:self.size]
        else:
            feed = feed[:self.size]
            
        return feed

    def _fetch_reliefweb_reports(self) -> List[Dict[str, Any]]:
        """Queries ReliefWeb disaster alerts API."""
        try:
            url = "https://api.reliefweb.int/v1/reports"
            params = {
                "appname": "aegissphere",
                "query[value]": "disaster OR earthquake OR flood OR wildfire OR hurricane",
                "limit": self.size,
                "fields[include][]": ["title", "body"]
            }
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            reports = res.json().get("data", [])
            
            feed = []
            for r in reports:
                fields = r.get("fields", {})
                title = fields.get("title", "")
                body = fields.get("body", "")
                text = f"{title}. {body[:300]}"
                
                # Determine label based on keywords
                text_lower = text.lower()
                if "earthquake" in text_lower or "seismic" in text_lower:
                    label = 0
                elif "flood" in text_lower or "deluge" in text_lower or "inundation" in text_lower:
                    label = 1
                elif "fire" in text_lower or "wildfire" in text_lower:
                    label = 2
                elif "hurricane" in text_lower or "cyclone" in text_lower or "typhoon" in text_lower:
                    label = 3
                else:
                    label = 4
                    
                feed.append({"text": text, "label": label})
            return feed
        except Exception as e:
            print(f"[ReliefWeb API Warning] Could not fetch live alerts: {e}. Utilizing local multilingual emergency news feed.")
            return []

    def _generate_multilingual_synthetics(self) -> List[Dict[str, Any]]:
        """High-fidelity multilingual news dataset covering major disaster types."""
        # Class mapping: 0: Earthquake, 1: Flood, 2: Wildfire, 3: Hurricane, 4: Other/None
        return [
            # English
            {"text": "A severe 6.8 magnitude earthquake struck off the coast of Sumatra today, triggering brief tsunami warnings.", "label": 0},
            {"text": "Heavy seasonal monsoons have led to catastrophic flash floods across several provinces, forcing mass evacuations.", "label": 1},
            {"text": "Extremely dry vegetation and high winds continue to fuel a massive wildfire spreading rapidly through the canyon.", "label": 2},
            {"text": "Category 4 Hurricane Milton has made landfall with sustained winds of 140 mph, causing massive storm surges.", "label": 3},
            {"text": "The local community holds a charity fundraising dinner to support climate resilience initiatives in urban sectors.", "label": 4},
            
            # Spanish
            {"text": "Un fuerte terremoto de magnitud 7.2 sacudió la región central del país, derrumbando múltiples estructuras.", "label": 0},
            {"text": "Las inundaciones repentinas causadas por las lluvias torrenciales han sumergido las principales autopistas.", "label": 1},
            {"text": "Incendio forestal fuera de control avanza rápidamente debido a la intensa ola de calor y vientos secos.", "label": 2},
            {"text": "El huracán categoría 5 se aproxima a la costa caribeña con vientos destructivos y marejadas ciclónicas.", "label": 3},
            {"text": "Reunión de ministros extranjeros se centra en nuevas políticas de sostenibilidad urbana para combatir el cambio climático.", "label": 4},
            
            # French
            {"text": "Un séisme de magnitude 5,9 a secoué les Alpes-Maritimes ce matin, provoquant des fissures sur les bâtiments.", "label": 0},
            {"text": "Des inondations sans précédent touchent le nord du pays après des pluies diluviennes ininterrompues.", "label": 1},
            {"text": "Un violent incendie de forêt a détruit plus de 2000 hectares de pinède sous l'effet du vent violent.", "label": 2},
            {"text": "Le cyclone tropical Belal frappe l'île avec des rafales de vent extrêmes et d'importantes précipitations.", "label": 3},
            {"text": "La conférence internationale sur la biodiversité s'est ouverte ce matin à Paris pour de nouveaux accords.", "label": 4},
            
            # Hindi
            {"text": "उत्तरी हिमालयी क्षेत्र में आज सुबह 6.2 तीव्रता का तेज भूकंप का झटका महसूस किया गया, लोग घरों से बाहर भागे।", "label": 0},
            {"text": "लगातार भारी बारिश के बाद प्रमुख नदियां खतरे के निशान से ऊपर बह रही हैं, जिससे बाढ़ का भारी खतरा पैदा हो गया है।", "label": 1},
            {"text": "भीषण गर्मी के बीच जंगलों में लगी आग बेकाबू हो गई है और अब वह रिहायशी इलाकों की तरफ बढ़ रही है।", "label": 2},
            {"text": "भयानक चक्रवाती तूफान बंगाल की खाड़ी से टकराया, तटीय क्षेत्रों में भारी तबाही और तेज हवाएं जारी हैं।", "label": 3},
            {"text": "कृषि मंत्रालय ने किसानों के लिए जैविक खेती को बढ़ावा देने हेतु नई प्रोत्साहन योजनाओं की घोषणा की है।", "label": 4},
            
            # Chinese
            {"text": "今日凌晨四川地区发生5.8级地震，造成部分震中房屋倒塌，救援人员正紧急赶往现场。", "label": 0},
            {"text": "由于连续三天的特大暴雨，多地河流发生漫堤，引发大面积洪涝和泥石流灾害。", "label": 1},
            {"text": "高温大风天气助长了山火蔓延，林业部门已出动直升机进行空中灭火作业。", "label": 2},
            {"text": "强台风在沿海地区登陆，最大风力达16级，已造成大范围停电和树木倒伏。", "label": 3},
            {"text": "全球科技巨头今天发布了新一代智能计算框架，旨在提高大型气象预测效率。", "label": 4}
        ]

class MockTokenizer:
    """Fallback basic tokenizer if HuggingFace tokenizers is unavailable."""
    def __call__(self, text: str, max_length: int = 128, padding: str = "max_length", truncation: bool = True, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        # Simple word splitting and converting to basic mock ascii ids
        words = text.lower().split()
        input_ids = [min(ord(c), 29999) for word in words for c in word][:max_length]
        
        # Pad sequence
        if len(input_ids) < max_length:
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids = input_ids + [0] * (max_length - len(input_ids))
        else:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * max_length
            
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
        }

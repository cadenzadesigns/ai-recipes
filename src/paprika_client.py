import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

from .models import Recipe


class PaprikaClient:
    """Client for interacting with Paprika Recipe Manager API.

    Uses v2 API with macOS client headers for authentication.

    IMPORTANT: As of 2024/2025, uploaded recipes don't sync to Paprika apps.
    The API accepts uploads but they don't appear in the recipe list or apps.
    This appears to be a limitation of the unofficial API.

    Technical details:
    - v2 authentication requires specific client headers (we use macOS)
    - Uploads return success (200 OK, {"result":true})
    - Uploaded recipes can be retrieved individually by UID
    - But recipes don't appear in the sync list (/sync/recipes/)
    - Sync status shows more recipes than the API returns (2400+ vs 865)

    See: https://gist.github.com/mattdsteele/7386ec363badfdeaad05a418b9a1f30a
    """

    BASE_URL = "https://www.paprikaapp.com/api/v2"
    LOGIN_URL = "https://www.paprikaapp.com/api/v2/account/login"

    def __init__(self, email: str = None, password: str = None):
        self.email = email or os.getenv("PAPRIKA_EMAIL")
        self.password = password or os.getenv("PAPRIKA_PASSWORD")

        if not self.email or not self.password:
            raise ValueError(
                "Paprika credentials must be provided or set in PAPRIKA_EMAIL and PAPRIKA_PASSWORD environment variables"
            )

        # Get v2 token
        self.token = self._authenticate()

    def _authenticate(self) -> str:
        """Authenticate with v2 API and get token."""
        headers = {
            "User-Agent": "Paprika Recipe Manager/3.0 (Macintosh; macOS 14.0)",
            "X-Paprika-Client": "macOS",
            "X-Paprika-Version": "3.0",
        }

        form_data = {"email": self.email, "password": self.password}

        response = requests.post(self.LOGIN_URL, data=form_data, headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"Failed to authenticate: {response.status_code} - {response.text}"
            )

        data = response.json()
        if "error" in data:
            raise ValueError(f"Authentication error: {data['error']['message']}")

        return data["result"]["token"]

    def _make_request(
        self, method: str, endpoint: str, data: Dict = None
    ) -> requests.Response:
        """Make authenticated request to Paprika API."""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        url = f"{self.BASE_URL}/{endpoint}"

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        return response

    def upload_recipe(self, recipe: Recipe) -> Dict:
        """Upload a recipe to Paprika."""
        # Generate a unique ID for the recipe
        recipe_uid = str(uuid.uuid4()).upper()

        # Convert our Recipe model to Paprika's format
        paprika_recipe = {
            "uid": recipe_uid,
            "name": recipe.name,
            "servings": recipe.servings or "",
            "source": recipe.source or "",
            "source_url": "",
            "notes": "\n\n".join(recipe.notes) if recipe.notes else "",
            "directions": "\n\n".join(recipe.directions),
            "ingredients": "\n".join(recipe.ingredients),
            "description": recipe.description or "",
            "total_time": recipe.total_time or "",
            "cook_time": "",
            "prep_time": "",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "photo_url": "",
            "photo": "",
            "photo_large": "",
            "photo_hash": "",
            "scale": "",
            "difficulty": "",
            "rating": 0,
            "on_favorites": False,
            "in_trash": False,
            "categories": [],
        }

        # Calculate hash as shown in the gist
        hash_dict = paprika_recipe.copy()
        hash_dict.pop("hash", None)  # Remove hash field if it exists
        recipe_hash = hashlib.sha256(
            json.dumps(hash_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()

        paprika_recipe["hash"] = recipe_hash

        # Upload via v2 API
        response = self._make_request(
            "POST", f"sync/recipe/{recipe_uid}/", paprika_recipe
        )

        if response.status_code not in (200, 201):
            raise ValueError(
                f"Failed to upload recipe: {response.status_code} - {response.text}"
            )

        try:
            return response.json()
        except Exception:
            # Some Paprika endpoints return text instead of JSON
            return {"status": "success", "response": response.text, "uid": recipe_uid}

    def get_recipes(self) -> list:
        """Get list of all recipes from Paprika."""
        response = self._make_request("GET", "sync/recipes/")

        if response.status_code != 200:
            raise ValueError(
                f"Failed to get recipes: {response.status_code} - {response.text}"
            )

        data = response.json()
        # v1 API returns {"result": [{"uid": "...", "hash": "..."}, ...]}
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    def get_recipe_details(self, uid: str) -> Optional[Dict]:
        """Get full recipe details by UID."""
        response = self._make_request("GET", f"sync/recipe/{uid}/")

        if response.status_code != 200:
            return None

        data = response.json()
        # v1 API returns {"result": {...}}
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    def search_recipe(self, name: str) -> Optional[Dict]:
        """Search for a recipe by name."""
        try:
            recipes = self.get_recipes()

            # For v1 API, we need to fetch each recipe's details to check the name
            for recipe_meta in recipes:
                if isinstance(recipe_meta, dict) and "uid" in recipe_meta:
                    recipe = self.get_recipe_details(recipe_meta["uid"])
                    if recipe and recipe.get("name", "").lower() == name.lower():
                        return recipe
        except Exception:
            # If search fails, we'll just proceed with upload
            pass

        return None

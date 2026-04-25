"""Customer (contact) tools — sevdesk /Contact API."""

from __future__ import annotations

from typing import Literal, Optional

import httpx

from app.client import get_client

# sevdesk category IDs
_CATEGORY = {
    "customer": 3,
    "supplier": 2,
    "partner": 4,
    "prospect": 28,
}


def _normalize_contact(c: dict) -> dict:
    name = c.get("name") or f"{c.get('surename', '')} {c.get('familyname', '')}".strip()
    cat = c.get("category") or {}
    cat_id = int(cat.get("id", 0)) if cat.get("id") else None
    contact_type = "person" if c.get("surename") else "company"
    return {
        "id": c.get("id"),
        "name": name,
        "type": contact_type,
        "customer_number": c.get("customerNumber"),
        "category_id": cat_id,
        "description": c.get("description"),
    }


async def list_customers(
    limit: int = 50,
    offset: int = 0,
    name: Optional[str] = None,
    category: Literal["customer", "supplier", "partner", "prospect"] = "customer",
) -> dict:
    """Lists contacts from sevdesk.

    Args:
        limit: Max records to return. Defaults to 50.
        offset: Pagination offset. Defaults to 0.
        name: Case-insensitive substring filter on contact name.
        category: Contact category to filter by. Defaults to 'customer'.

    Returns:
        Dict with total, offset, and a list of contact records.
    """
    params: dict = {
        "limit": limit,
        "offset": offset,
        "depth": 0,
        "category[id]": _CATEGORY[category],
        "category[objectName]": "Category",
    }
    if name:
        params["name"] = name

    try:
        resp = await get_client().get("/Contact", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    return {
        "total": data.get("total", len(objects)),
        "offset": offset,
        "customers": [_normalize_contact(c) for c in objects],
    }


async def get_customer(contact_id: str) -> dict:
    """Gets detailed information about a single contact by ID.

    Args:
        contact_id: The sevdesk contact ID.

    Returns:
        Full contact record, or an error dict if not found.
    """
    try:
        resp = await get_client().get(f"/Contact/{contact_id}", params={"depth": 1})
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Contact '{contact_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    if not objects:
        return {"error": f"Contact '{contact_id}' not found."}

    c = objects[0] if isinstance(objects, list) else objects
    normalized = _normalize_contact(c)

    # Pull communication ways (email, phone) from the nested array
    comm_ways = c.get("communicationWays") or []
    for cw in comm_ways:
        if cw.get("type") == "EMAIL":
            normalized["email"] = cw.get("value")
        elif cw.get("type") == "PHONE":
            normalized["phone"] = cw.get("value")

    # Pull primary address
    addresses = c.get("addresses") or []
    if addresses:
        addr = addresses[0]
        normalized["street"] = addr.get("street")
        normalized["city"] = addr.get("city")
        normalized["zipcode"] = addr.get("zip")
        normalized["country"] = (addr.get("country") or {}).get("code")

    return normalized


async def create_customer(
    name: str,
    type: Literal["company", "person"] = "company",
    category: Literal["customer", "supplier", "partner", "prospect"] = "customer",
    customer_number: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Creates a new contact in sevdesk.

    Args:
        name: Company name (for organizations) or surname (for individuals).
        type: Contact type — 'company' or 'person'. Defaults to 'company'.
        category: Contact category. Defaults to 'customer'.
        customer_number: Optional customer number (auto-generated if omitted).
        description: Optional contact description or notes.

    Returns:
        The newly created contact record, or an error dict.
    """
    payload: dict = {
        "objectName": "Contact",
        "category": {"id": _CATEGORY[category], "objectName": "Category"},
    }
    if type == "company":
        payload["name"] = name
    else:
        # sevdesk splits person names into surename / familyname
        parts = name.rsplit(" ", 1)
        payload["surename"] = parts[0]
        payload["familyname"] = parts[1] if len(parts) > 1 else ""

    if customer_number:
        payload["customerNumber"] = customer_number
    if description:
        payload["description"] = description

    try:
        resp = await get_client().post("/Contact", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    c = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    return _normalize_contact(c)


async def edit_customer(
    contact_id: str,
    name: Optional[str] = None,
    customer_number: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Updates an existing contact in sevdesk.

    Only provided fields are updated; omitted fields remain unchanged.

    Args:
        contact_id: The sevdesk contact ID to update.
        name: Updated company/contact name.
        customer_number: Updated customer number.
        description: Updated description or notes.

    Returns:
        The updated contact record, or an error dict.
    """
    payload: dict = {"objectName": "Contact"}
    if name is not None:
        payload["name"] = name
    if customer_number is not None:
        payload["customerNumber"] = customer_number
    if description is not None:
        payload["description"] = description

    if len(payload) == 1:
        return {"error": "No fields provided to update."}

    try:
        resp = await get_client().put(f"/Contact/{contact_id}", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Contact '{contact_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    c = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    return _normalize_contact(c)

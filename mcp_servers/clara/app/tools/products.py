"""Product tools — sevdesk /Part API."""

from __future__ import annotations

from typing import Optional

import httpx

from app.client import get_client

_UNITY_ID = "1"  # sevdesk unity id 1 = Stück (pcs)


def _normalize_part(p: dict) -> dict:
    unity = p.get("unity") or {}
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "description": p.get("text"),
        "product_no": p.get("partNumber"),
        "unit": unity.get("name") if isinstance(unity, dict) else None,
        "unit_price": float(p.get("price") or 0),
        "tax_rate": float(p.get("taxRate") or 0),
        "stock": p.get("stockCount"),
    }


async def list_products(
    limit: int = 100,
    offset: int = 0,
    name: Optional[str] = None,
) -> dict:
    """Lists products (parts) from sevdesk.

    Args:
        limit: Max records to return. Defaults to 100.
        offset: Pagination offset. Defaults to 0.
        name: Substring filter on product name.

    Returns:
        Dict with total, offset, and a list of product records.
    """
    params: dict = {"limit": limit, "offset": offset}
    if name:
        params["name"] = name

    try:
        resp = await get_client().get("/Part", params=params)
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
        "products": [_normalize_part(p) for p in objects],
    }


async def get_product(product_id: str) -> dict:
    """Gets detailed information about a single product by ID.

    Args:
        product_id: The sevdesk part ID.

    Returns:
        Full product record, or an error dict if not found.
    """
    try:
        resp = await get_client().get(f"/Part/{product_id}")
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Product '{product_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or []
    if not objects:
        return {"error": f"Product '{product_id}' not found."}
    p = objects[0] if isinstance(objects, list) else objects
    return _normalize_part(p)


async def create_product(
    name: str,
    unit_price: float,
    description: Optional[str] = None,
    tax_rate: float = 19.0,
    product_no: Optional[str] = None,
) -> dict:
    """Creates a new product in sevdesk.

    Args:
        name: Product name.
        unit_price: Unit price (excl. tax).
        description: Optional product description.
        tax_rate: Tax rate percentage. Defaults to 19.0.
        product_no: Optional product/article number.

    Returns:
        The newly created product record, or an error dict.
    """
    payload: dict = {
        "objectName": "Part",
        "name": name,
        "price": unit_price,
        "taxRate": tax_rate,
        "unity": {"id": _UNITY_ID, "objectName": "Unity"},
    }
    if description:
        payload["text"] = description
    if product_no:
        payload["partNumber"] = product_no

    try:
        resp = await get_client().post("/Part", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    p = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    return _normalize_part(p)


async def edit_product(
    product_id: str,
    name: Optional[str] = None,
    unit_price: Optional[float] = None,
    description: Optional[str] = None,
    product_no: Optional[str] = None,
) -> dict:
    """Updates an existing product in sevdesk.

    Only provided fields are updated; omitted fields remain unchanged.

    Args:
        product_id: The sevdesk part ID to update.
        name: Updated product name.
        unit_price: Updated unit price (excl. tax).
        description: Updated description.
        product_no: Updated product/article number.

    Returns:
        The updated product record, or an error dict.
    """
    payload: dict = {"objectName": "Part"}
    if name is not None:
        payload["name"] = name
    if unit_price is not None:
        payload["price"] = unit_price
    if description is not None:
        payload["text"] = description
    if product_no is not None:
        payload["partNumber"] = product_no

    if len(payload) == 1:
        return {"error": "No fields provided to update."}

    try:
        resp = await get_client().put(f"/Part/{product_id}", json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"error": f"Product '{product_id}' not found."}
        return {"error": f"API error {e.response.status_code}: {e.response.text}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {e}"}

    objects = data.get("objects") or {}
    p = objects if isinstance(objects, dict) else (objects[0] if objects else {})
    return _normalize_part(p)

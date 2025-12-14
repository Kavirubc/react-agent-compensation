"""Family Coordination MCP Server with Compensation Annotations.

This MCP server demonstrates compensation patterns with 19 tools for
coordinating family activities. Each tool that creates resources has a
corresponding compensation tool to undo/rollback the action.

Compensation pairs are declared via x-compensation-pair annotations:
- add_family_member <-> delete_family_member
- add_task <-> delete_task
- add_pickup <-> delete_pickup
- add_cooking_task <-> delete_cooking_task

Additional metadata:
- x-action-type: "create", "delete", "update", "read"
- x-reversible: True for update operations that store previous state
- x-destructive: True for dangerous operations (reset_schedule)

Run:
    python server.py

Requires:
    - MongoDB on localhost:27017
    - pip install fastmcp pymongo python-dotenv
"""

from bson import ObjectId
from fastmcp import FastMCP

from examples.mcp.database import (
    cooking_tasks,
    family_members,
    pickups,
    tasks,
)

mcp = FastMCP("Family Coordination MCP Server")


def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


# ============== FAMILY MEMBER TOOLS ==============


@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_family_member",
        "x-category": "family",
        "x-action-type": "create",
    }
)
def add_family_member(
    name: str,
    role: str,
    arrival_method: str = None,
    arrival_time: str = None,
    location: str = None,
) -> dict:
    """Add a family member to the coordination system.

    Args:
        name: Name of the family member (e.g., Sarah, James)
        role: Role in the family (e.g., Mom, Dad, Sister, Brother, Grandma)
        arrival_method: How they're arriving (e.g., flight, driving, at_home, needs_pickup)
        arrival_time: Expected arrival time (e.g., 13:00, 14:30)
        location: Current or starting location (e.g., SF, Chicago, NY, suburban Boston)
    """
    member = {
        "name": name,
        "role": role,
        "arrival_method": arrival_method,
        "arrival_time": arrival_time,
        "location": location,
        "status": "not_arrived",
    }
    result = family_members.insert_one(member)
    return {"id": str(result.inserted_id), "name": name, "message": f"Added family member: {name}"}


@mcp.tool(
    annotations={
        "x-compensation-pair": "add_family_member",
        "x-category": "family",
        "x-action-type": "delete",
    }
)
def delete_family_member(name: str) -> dict:
    """Delete a family member from the coordination system.

    Args:
        name: Name of the family member to delete
    """
    # Store for potential recreation
    member = family_members.find_one({"name": name})
    result = family_members.delete_one({"name": name})
    if result.deleted_count == 0:
        return {"error": f"Member {name} not found"}
    return {
        "message": f"Deleted family member: {name}",
        "deleted_member": serialize_doc(member) if member else None,
    }


@mcp.tool(annotations={"x-category": "family", "x-action-type": "read"})
def get_family_members() -> list:
    """Get all family members and their current status."""
    return [serialize_doc(m) for m in family_members.find()]


@mcp.tool(
    annotations={"x-category": "family", "x-action-type": "update", "x-reversible": True}
)
def update_family_member_status(name: str, status: str, location: str = None) -> dict:
    """Update a family member's status.

    Args:
        name: Name of the family member
        status: New status (e.g., not_arrived, en_route, arrived, at_home)
        location: Current location if applicable
    """
    previous = family_members.find_one({"name": name})
    if not previous:
        return {"error": f"Member {name} not found"}

    update = {"status": status}
    if location:
        update["location"] = location

    family_members.update_one({"name": name}, {"$set": update})
    return {
        "message": f"Updated {name}'s status to {status}",
        "previous_status": previous.get("status"),
        "previous_location": previous.get("location"),
    }


# ============== TASK TOOLS ==============


@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_task",
        "x-category": "tasks",
        "x-action-type": "create",
    }
)
def add_task(
    task_name: str,
    assigned_to: str = None,
    start_time: str = None,
    end_time: str = None,
    notes: str = None,
) -> dict:
    """Add a general task to the schedule.

    Args:
        task_name: Description of the task
        assigned_to: Person responsible for this task
        start_time: When the task should start (e.g., 14:00)
        end_time: When the task should end (e.g., 15:00)
        notes: Additional notes about the task
    """
    task = {
        "task_name": task_name,
        "assigned_to": assigned_to,
        "status": "pending",
        "start_time": start_time,
        "end_time": end_time,
        "notes": notes,
    }
    result = tasks.insert_one(task)
    return {"id": str(result.inserted_id), "task_name": task_name, "message": f"Added task: {task_name}"}


@mcp.tool(
    annotations={
        "x-compensation-pair": "add_task",
        "x-category": "tasks",
        "x-action-type": "delete",
    }
)
def delete_task(task_id: str) -> dict:
    """Delete a task from the schedule.

    Args:
        task_id: The ID of the task to delete
    """
    task = tasks.find_one({"_id": ObjectId(task_id)})
    if not task:
        return {"error": "Task not found"}

    tasks.delete_one({"_id": ObjectId(task_id)})
    return {"message": "Task deleted", "deleted_task": serialize_doc(task)}


@mcp.tool(annotations={"x-category": "tasks", "x-action-type": "read"})
def get_tasks() -> list:
    """Get all tasks in the schedule."""
    return [serialize_doc(t) for t in tasks.find()]


@mcp.tool(
    annotations={"x-category": "tasks", "x-action-type": "update", "x-reversible": True}
)
def update_task_status(task_id: str, status: str) -> dict:
    """Update a task's status (pending, in_progress, completed).

    Args:
        task_id: The ID of the task to update
        status: New status (pending, in_progress, completed)
    """
    previous = tasks.find_one({"_id": ObjectId(task_id)})
    if not previous:
        return {"error": "Task not found"}

    tasks.update_one({"_id": ObjectId(task_id)}, {"$set": {"status": status}})
    return {
        "message": f"Task status updated to {status}",
        "previous_status": previous.get("status"),
    }


# ============== PICKUP TOOLS ==============


@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_pickup",
        "x-category": "pickups",
        "x-action-type": "create",
    }
)
def add_pickup(
    person_to_pickup: str,
    pickup_location: str,
    driver: str = None,
    pickup_time: str = None,
) -> dict:
    """Add a pickup task.

    Args:
        person_to_pickup: Who needs to be picked up
        pickup_location: Where to pick them up (e.g., BOS Airport, suburban Boston)
        driver: Who will drive to pick them up
        pickup_time: When to pick them up (e.g., 14:30)
    """
    pickup = {
        "person_to_pickup": person_to_pickup,
        "pickup_location": pickup_location,
        "driver": driver,
        "pickup_time": pickup_time,
        "status": "pending",
    }
    result = pickups.insert_one(pickup)
    return {
        "id": str(result.inserted_id),
        "person_to_pickup": person_to_pickup,
        "message": f"Added pickup for {person_to_pickup}",
    }


@mcp.tool(
    annotations={
        "x-compensation-pair": "add_pickup",
        "x-category": "pickups",
        "x-action-type": "delete",
    }
)
def delete_pickup(pickup_id: str) -> dict:
    """Delete a pickup task.

    Args:
        pickup_id: The ID of the pickup to delete
    """
    pickup = pickups.find_one({"_id": ObjectId(pickup_id)})
    if not pickup:
        return {"error": "Pickup not found"}

    pickups.delete_one({"_id": ObjectId(pickup_id)})
    return {"message": "Pickup deleted", "deleted_pickup": serialize_doc(pickup)}


@mcp.tool(annotations={"x-category": "pickups", "x-action-type": "read"})
def get_pickups() -> list:
    """Get all pickup tasks."""
    return [serialize_doc(p) for p in pickups.find()]


@mcp.tool(
    annotations={
        "x-category": "pickups",
        "x-action-type": "update",
        "x-reversible": True,
    }
)
def update_pickup(
    pickup_id: str,
    driver: str = None,
    pickup_time: str = None,
    status: str = None,
) -> dict:
    """Update a pickup task.

    Args:
        pickup_id: The ID of the pickup to update
        driver: Update the driver
        pickup_time: Update the pickup time
        status: Update status (pending, in_progress, completed)
    """
    previous = pickups.find_one({"_id": ObjectId(pickup_id)})
    if not previous:
        return {"error": "Pickup not found"}

    update = {}
    if driver:
        update["driver"] = driver
    if pickup_time:
        update["pickup_time"] = pickup_time
    if status:
        update["status"] = status
    if not update:
        return {"error": "No updates provided"}

    pickups.update_one({"_id": ObjectId(pickup_id)}, {"$set": update})
    return {
        "message": "Pickup updated",
        "previous_driver": previous.get("driver"),
        "previous_pickup_time": previous.get("pickup_time"),
        "previous_status": previous.get("status"),
    }


# ============== COOKING TOOLS ==============


@mcp.tool(
    annotations={
        "x-compensation-pair": "delete_cooking_task",
        "x-category": "cooking",
        "x-action-type": "create",
    }
)
def add_cooking_task(
    dish_name: str,
    cook_time_hours: float,
    start_time: str = None,
    supervisor: str = None,
) -> dict:
    """Add a cooking task.

    Args:
        dish_name: Name of the dish (e.g., Turkey, Side dishes)
        cook_time_hours: How long it takes to cook in hours (e.g., 4.0 for turkey)
        start_time: When to start cooking (e.g., 14:00)
        supervisor: Who will supervise the cooking (required for fire safety)
    """
    cooking = {
        "dish_name": dish_name,
        "cook_time_hours": cook_time_hours,
        "start_time": start_time,
        "supervisor": supervisor,
        "status": "pending",
    }
    result = cooking_tasks.insert_one(cooking)
    return {
        "id": str(result.inserted_id),
        "dish_name": dish_name,
        "message": f"Added cooking task: {dish_name}",
    }


@mcp.tool(
    annotations={
        "x-compensation-pair": "add_cooking_task",
        "x-category": "cooking",
        "x-action-type": "delete",
    }
)
def delete_cooking_task(task_id: str) -> dict:
    """Delete a cooking task.

    Args:
        task_id: The ID of the cooking task to delete
    """
    task = cooking_tasks.find_one({"_id": ObjectId(task_id)})
    if not task:
        return {"error": "Cooking task not found"}

    cooking_tasks.delete_one({"_id": ObjectId(task_id)})
    return {"message": "Cooking task deleted", "deleted_task": serialize_doc(task)}


@mcp.tool(annotations={"x-category": "cooking", "x-action-type": "read"})
def get_cooking_tasks() -> list:
    """Get all cooking tasks."""
    return [serialize_doc(c) for c in cooking_tasks.find()]


@mcp.tool(
    annotations={
        "x-category": "cooking",
        "x-action-type": "update",
        "x-reversible": True,
    }
)
def update_cooking_task(
    task_id: str,
    start_time: str = None,
    supervisor: str = None,
    status: str = None,
) -> dict:
    """Update a cooking task.

    Args:
        task_id: The ID of the cooking task
        start_time: Update start time
        supervisor: Update supervisor
        status: Update status (pending, in_progress, completed)
    """
    previous = cooking_tasks.find_one({"_id": ObjectId(task_id)})
    if not previous:
        return {"error": "Cooking task not found"}

    update = {}
    if start_time:
        update["start_time"] = start_time
    if supervisor:
        update["supervisor"] = supervisor
    if status:
        update["status"] = status
    if not update:
        return {"error": "No updates provided"}

    cooking_tasks.update_one({"_id": ObjectId(task_id)}, {"$set": update})
    return {
        "message": "Cooking task updated",
        "previous_start_time": previous.get("start_time"),
        "previous_supervisor": previous.get("supervisor"),
        "previous_status": previous.get("status"),
    }


# ============== SCHEDULE TOOLS ==============


@mcp.tool(annotations={"x-category": "schedule", "x-action-type": "read"})
def get_full_schedule() -> dict:
    """Get the complete schedule with all family members, tasks, pickups, and cooking tasks."""
    return {
        "dinner_time": "18:00",
        "family_members": [serialize_doc(m) for m in family_members.find()],
        "tasks": [serialize_doc(t) for t in tasks.find()],
        "pickups": [serialize_doc(p) for p in pickups.find()],
        "cooking_tasks": [serialize_doc(c) for c in cooking_tasks.find()],
    }


@mcp.tool(
    annotations={
        "x-category": "schedule",
        "x-action-type": "delete",
        "x-destructive": True,
        "x-requires-confirmation": True,
    }
)
def reset_schedule() -> dict:
    """Reset the entire schedule - clears all data. This is destructive and cannot be undone."""
    family_members.delete_many({})
    tasks.delete_many({})
    pickups.delete_many({})
    cooking_tasks.delete_many({})
    return {"message": "Schedule reset - all data cleared"}


@mcp.tool(annotations={"x-category": "reference", "x-action-type": "read"})
def get_travel_times() -> dict:
    """Get travel time constraints for planning."""
    return {
        "home_to_bos_airport": "60 minutes",
        "bos_airport_to_grandmas": "60 minutes",
        "home_to_grandmas": "30 minutes",
    }


if __name__ == "__main__":
    print("Starting Family Coordination MCP Server on http://0.0.0.0:8000/sse")
    print("\nCompensation pairs:")
    print("  add_family_member <-> delete_family_member")
    print("  add_task <-> delete_task")
    print("  add_pickup <-> delete_pickup")
    print("  add_cooking_task <-> delete_cooking_task")
    print("\nRequires MongoDB on localhost:27017")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)

class Task:
    def __init__(self, title, description, due_date, status="Incomplete"):
        self.title = title
        self.description = description
        self.due_date = due_date
        self.status = status

    def mark_complete(self):
        self.status = "Complete"

    def __str__(self):
        return f"Title: {self.title}\nDescription: {self.description}\nDue Date: {self.due_date}\nStatus: {self.status}\n"
    
    
class ToDoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def mark_task_complete(self, title):
        for task in self.tasks:
            if task.title == title:
                task.mark_complete()
                return 
        print(f"Task '{title}' not found.")

    def list_all_tasks(self):
        if not self.tasks:
            print("No tasks in the list.")
            return

        for task in self.tasks:
            print(task)

    def list_incomplete_tasks(self):
        incomplete_tasks = [task for task in self.tasks if task.status == "Incomplete"]
        if not incomplete_tasks:
            print("No incomplete tasks.")
            return

        for task in incomplete_tasks:
            print(task)
            
def main():
    todo_list = ToDoList()

    while True:
        print("\n--- ToDo List Application ---")
        print("1. Add Task")
        print("2. Mark Task as Complete")
        print("3. List All Tasks")
        print("4. List Incomplete Tasks")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            title = input("Enter task title: ")
            description = input("Enter task description: ")
            due_date = input("Enter due date: ")
            task = Task(title, description, due_date)
            todo_list.add_task(task)
            print("Task added!")

        elif choice == "2":
            title = input("Enter task title to mark as complete: ")
            todo_list.mark_task_complete(title)

        elif choice == "3":
            todo_list.list_all_tasks()

        elif choice == "4":
            todo_list.list_incomplete_tasks()

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
from flask_script import Manager, Server
from app import myapp

print('manage.py')
manager = Manager(myapp)
server = Server(host='localhost', port=8080, use_reloader=True, use_debugger=True)
manager.add_command("runserver", server)

if __name__ == '__main__':
    print('manager.run()')
    manager.run()
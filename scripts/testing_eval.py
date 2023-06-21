class MyClass:
    def __init__(self):
        self.test = "test"
        eval("print(self.test)")


inst = MyClass()

def add(x, value):
    return x + value

print(add(3, 2))
print(add(0, 1))

def item_operation(shopping_cart, operation, name, price):
    if operation == 'add':
        shopping_cart.append((name, price))
        return shopping_cart
    elif operation == 'remove':
        shopping_cart.remove((name, price))
        return shopping_cart
    else:
        return shopping_cart

def total_price(shopping_cart, vat):
    return sum([item[1] for item in shopping_cart]) * (1 + vat)

shopping_cart = []
shopping_cart = item_operation(shopping_cart, 'add', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'add', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'remove', 'milk', 0.79)
shopping_cart = item_operation(shopping_cart, 'add', 'butter', 1.49)
shopping_cart = item_operation(shopping_cart, 'add', 'eggs', 2.89)
total_price = total_price(shopping_cart, 0.20)

print(shopping_cart, total_price)


class ShoppingCart(object):

    def __init__(self, vat):
        self.contents = []
        self.vat = vat

    def add(self, name, price):
        self.contents.append((name, price))

    def remove(self, name, price):
        self.contents.remove((name, price))

    def total_price(self):
        return sum([item[1] for item in self.contents]) * (1 + self.vat)


cart = ShoppingCart(vat=0.20)
cart.add('milk', 0.79)
cart.add('milk', 0.79)
cart.remove('milk', 0.79)
cart.add('butter', 1.49)
cart.add('eggs', 2.89)

print(cart.contents, cart.total_price())

'''
Classes have this special __init__ function, which is called when the class is initialized 
(or instantiated) with ShoppingCart(). This is typically the place to define what are 
called class 'attributes'; class-wide variables we can access in its methods (functions) 
through self (which points to the class / object itself). All regular class method definitions 
therefore start with this (def add(self, ). These attributes can also be accessed outside 
of the class (see cart.contents).

Classes can also inherit functionality from each other, like so:
'''


class A(object):

    def __init__(self):
        self.x = 2

    def add(self):
        return self.x + 1


class B(A):

    def add(self):
        return self.x + 2


class C(A):

    def __init__(self):
        self.x = 6

    def something(self):
        return None


a, b, c = A(), B(), C()
print(a.add(), b.add(), c.add())


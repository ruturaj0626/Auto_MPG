from abc import ABC, abstractmethod

class Coffee(ABC):
    """Abstract Coffee Class"""

    @abstractmethod
    def prepare(self):
        pass

class Espresso(Coffee):
    """Concrete Espresso Class"""

    def prepare(self):
        print("Grinding espresso beans...")
        print("Boiling water...")
        print("Pouring hot water over espresso grounds...")

class Cappuccino(Coffee):
    """Concrete Cappuccino Class"""

    def prepare(self):
        print("Grinding coffee beans...")
        print("Steaming milk...")
        print("Pouring espresso and steamed milk into a cup...")

class Latte(Coffee):
    """Concrete Latte Class"""

    def prepare(self):
        print("Grinding coffee beans...")
        print("Steaming milk...")
        print("Pouring espresso and steamed milk into a cup with foam...")

class CoffeeShop(ABC):
    """Abstract Coffee Shop Class"""

    @abstractmethod
    def order_coffee(self, coffee_type):
        pass

class ItalianCoffeeShop(CoffeeShop):
    """Concrete Italian Coffee Shop"""

    def order_coffee(self, coffee_type):
        if coffee_type == "espresso":
            return Espresso()
        elif coffee_type == "cappuccino":
            return Cappuccino()
        else:
            raise ValueError("Invalid coffee type")

class AmericanCoffeeShop(CoffeeShop):
    """Concrete American Coffee Shop"""

    def order_coffee(self, coffee_type):
        if coffee_type == "latte":
            return Latte()
        else:
            raise ValueError("Invalid coffee type")

if __name__ == "__main__":
    italian_shop = ItalianCoffeeShop()
    espresso = italian_shop.order_coffee("espresso")
    espresso.prepare()

    american_shop = AmericanCoffeeShop()
    latte = american_shop.order_coffee("latte")
    latte.prepare()
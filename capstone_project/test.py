class Car():
    def __init__(self, brand: str, colour: str, model: str):
        self.brand = brand
        self.colour = colour
        self.model = model
        self.position = 0
 
    def go_forward(self, num_of_steps: int) -> int:
        self.position = self.position + num_of_steps
        return self.position
    
    def go_backward(self, num_of_steps: int) -> int:
        self.position = self.position - num_of_steps

def describe_car(a_car: Car) -> dict:
    a_car_description = {
        "car_brand": a_car.brand,
        "car_model": a_car.model,
        "car_colour": a_car.colour}
    return a_car_description
    #return (f'{k}: {v}' for k, v in record.items()), sep='\n', end='\n\n')

if __name__== "__main__":

    toyota_yaris = Car(brand="Toyota", colour="purple", model="Yaris")


    print(toyota_yaris.colour)

    print(toyota_yaris.position)
    print(toyota_yaris.go_backward(num_of_steps=5))
    print(describe_car(a_car = toyota_yaris))



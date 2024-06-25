
class vehicle:
    def __init__(self, brand: str, model: str, color: str):
        self.brand = brand
        self.model = model
        self.color = color
        self.speed = 0
        self.position = 0

    def go_forward(self, num_of_steps: int) -> int:
        self.position = self.position + num_of_steps
        return self.position
    
    def go_backwards(self, num_of_steps: int) -> int:
        self.position - self. position - num_of_steps
        return self.position
    
    @classmethod
    def get_num_wheels(cls):
        return self.num_of_wheels
    
    def accelerate(self):
         pass
    
    def accelerate(self, speed: int) -> int:
         self.speed = self.speed + speed
         return self.speed

class Car():
    num_wheels = 4
    def __init__(self, brand: str, colour: str, model: str, max_km_hour: int):
        super().__init__(brand=brand, model=model, color=color)
      ## if user super remove below three
      #   self.brand = brand
      #  self.colour = colour
       # self.model = model
        self.max_km_hour = max_km_hour
        self.km_hour = 0

        def drive(self):
            return "driving on road!"
        
        def accelerate(self, km_hour: int) ->int:
            self.speed = self.speed + km_hour

class Boat (Vehicle):
        def __init__(self, brand: str, color: str, model: str, max_knot_hour: int):
            super().__init__(brand = brand, model = model, color = color)
            self.max_knot_hr = max_knot_hr
            self.knot_hour = 0

        def sail(self) -> str:
             return "sailing!"
        
        def accelerate(self, knot_hour: int) ->int:
            self.speed = self.speed + knot_hour
 
    def go_forward(self, num_of_steps: int) -> int:
        self.position = self.position + num_of_steps
        return self.position
    
    def go_backward(self, num_of_steps: int) -> int:
        self.position = self.position - num_of_steps
        return self.position
    


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



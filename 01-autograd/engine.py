class Value:

    # The constructor
    def __init__(self, num):
        self.num = num
        self.prod_vals = []
        self.grad = 0
        self.push_grads = lambda:None

    # Defining multiplication for the Value class
    def __mul__(self, other):
        # If we have a.__mul__(b) -> c, we wan't the c Value to store a and b   
        # as its prod vals (the values which produced c)
        
        c = Value(self.num * other.num)
        c.prod_vals = [self,other]
        
        # Setting up the push_grads function for c  
        def mul_push_grads(first_val,second_val):
            first_val.grad = second_val.num
            second_val.grad = first_val.num

        c.push_grads = mul_push_grads

        return c



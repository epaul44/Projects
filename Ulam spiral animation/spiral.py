import turtle


def is_prime(num):
    """
    Determines if a number is prime using math
    :param num: an int
    :return: a boolean, True if the number is prime, False if not
    """
    for factor in range(2, num // 2 + 1):
        if num % factor == 0:
            return False
    return True

# The line below determines static vs animated image
#turtle.tracer(0, 0)

# initialize
turtle.left(90)
turtle.penup()
counter = 0
side = 1

# start at 2 since 1 is not prime
for num in range(2, 10000):
    # moves tracer and draws based on number
    turtle.forward(5)
    if is_prime(num):
        turtle.dot(3, 'red')
    counter += 1

    # determines when to turn   
    if counter == side:
        counter = 0
        turtle.left(90)
        if turtle.heading() in (90, 270):
            side += 1

# use in conjuction with tracer
#turtle.update()
turtle.exitonclick()

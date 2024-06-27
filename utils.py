def is_food_in_snake(food, snake) -> bool:
    """
    Function which gives answer if generated food is located inside head or any of the body parts of the snake.
    """
    
    # Food in head
    if (snake.head.x == food.x and snake.head.y == food.y):
        return True
    
    # Food in body parts
    for body_part in snake.body:
        if body_part.x == food.x and body_part.y == food.y:
            print("Food generated in body!")
            return True
        
    return False

def check_for_collision(snake, SCREEN_WIDTH, SCREEN_HEIGHT):
    """
    Function which check whether collision occured, if snake went into itself or tried to cross the border.
    """
    
    head, body = snake.head, snake.body
    
    # Check for border collisionsd
    if head.x >= SCREEN_WIDTH or head.x < 0 or head.y >= SCREEN_HEIGHT or head.y < 0:
        return True
    # Check for self collision
    for body_part in body:
        if body_part.x == head.x and body_part.y == head.y:
            return True
    return False


def get_danger_state(snake):
    danger_up = check_for_collision_helper(snake, 0, -1)
    danger_down = check_for_collision_helper(snake, 0, 1)
    danger_left = check_for_collision_helper(snake, -1, 0)
    danger_right = check_for_collision_helper(snake, 1, 0)
    
    return [danger_up, danger_down, danger_left, danger_right]

def check_for_collision_helper(snake, x_offset, y_offset):
    new_head = snake.head.copy()
    new_head.x += x_offset * snake.head.width
    new_head.y += y_offset * snake.head.height

    if new_head.x < 0 or new_head.x >= snake.head.width * 10 or new_head.y < 0 or new_head.y >= snake.head.height * 10:
        return 1
    for part in snake.body:
        if new_head.colliderect(part):
            return 1
    return 0

def get_food_state(snake):
    food_up = 1 if snake.food.y < snake.head.y else 0
    food_down = 1 if snake.food.y > snake.head.y else 0
    food_left = 1 if snake.food.x < snake.head.x else 0
    food_right = 1 if snake.food.x > snake.head.x else 0
    
    return [food_up, food_down, food_left, food_right]
    
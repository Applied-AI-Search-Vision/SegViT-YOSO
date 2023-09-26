import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
DOT_SPEED = 7
OBSTACLE_SPEED = 8
NUM_EPISODES = 400
LR_DECAY = 0.995
MIN_LR = 0.0004

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 128, output_dim = 4, lr=0.0005):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x - torch.max(x)  # Subtracting the max for stability before softmax
        return self.softmax(x)

class Agent:
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4, lr=0.001):
        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.transitions = []
        self.gamma = 0.95
        self.epsilon = 1.5
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.09
        self.lr = lr

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Check for invalid inputs
        if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
            action = random.randint(0, 3)
            return action
        
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                probabilities = self.policy_network(state_tensor)
            action = torch.multinomial(probabilities, 1).item()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def store_transition(self, state, action, reward):
        self.transitions.append((state, action, reward))

    def train(self):
        if not self.transitions:
            return

        states, actions, rewards = zip(*self.transitions)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        discounted_rewards = [reward * (self.gamma ** i) for i, reward in enumerate(rewards)]

        self.optimizer.zero_grad()
        action_probs = self.policy_network(states)
        action_taken_probs = action_probs[range(len(actions)), actions]
        loss = -torch.sum(torch.log(action_taken_probs) * torch.FloatTensor(discounted_rewards))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(MIN_LR, param_group['lr'] * LR_DECAY)

        self.transitions = []

class Dot:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = DOT_SPEED
        self.prev_positions = [(self.x, self.y)]

    def move(self, action):
        if action == 0:
            self.x = max(10, self.x - self.speed)
        elif action == 1:
            self.x = min(WIDTH - 10, self.x + self.speed)
        elif action == 2:
            self.y = max(10, self.y - self.speed)
        elif action == 3:
            self.y = min(HEIGHT - 10, self.y + self.speed)

        self.prev_positions.append((self.x, self.y))
        self.prev_positions = self.prev_positions[-10:]

    def display(self):
        pygame.draw.circle(screen, GREEN, (self.x, self.y), 10)

class Obstacle:
    def __init__(self):
        self.width = 50
        self.height = 50
        self.speed = OBSTACLE_SPEED
        self.direction = random.choice([0, 1, 2, 3])

        if self.direction == 0:
            self.x = random.randint(0, WIDTH - self.width)
            self.y = 0
        elif self.direction == 1:
            self.x = 0
            self.y = random.randint(0, HEIGHT - self.height)
        elif self.direction == 2:
            self.x = WIDTH - self.width
            self.y = random.randint(0, HEIGHT - self.height)
        else:
            self.x = random.randint(0, WIDTH - self.width)
            self.y = HEIGHT - self.height

    def move(self):
        if self.direction == 0:
            self.y += self.speed
        elif self.direction == 1:
            self.x += self.speed
        elif self.direction == 2:
            self.x -= self.speed
        else:
            self.y -= self.speed

    def display(self):
        pygame.draw.rect(screen, RED, (self.x, self.y, self.width, self.height))

def main():
    agent_instance = Agent(output_dim=4)
    scores = []

    for episode in range(NUM_EPISODES):
        dot = Dot()
        obstacles = []
        score = 0

        running = True
        while running:
            screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            nearest_obstacle = obstacles[0] if obstacles else None
            nearest_obstacle_x = nearest_obstacle.x if nearest_obstacle else 0
            nearest_obstacle_y = nearest_obstacle.y if nearest_obstacle else 0

            state = [dot.x, dot.y, nearest_obstacle_x, nearest_obstacle_y]
            action = agent_instance.choose_action(state)
            dot.move(action)

            dot.display()

            if random.random() < 0.03:
                obstacles.append(Obstacle())

            collision = False
            for obstacle in obstacles:
                obstacle.move()
                obstacle.display()
                if obstacle.y > HEIGHT or obstacle.x > WIDTH or obstacle.x + obstacle.width < 0 or obstacle.y + obstacle.height < 0:
                    obstacles.remove(obstacle)
                    score += 1
                if (obstacle.x < dot.x < obstacle.x + obstacle.width or obstacle.x < dot.x + 10 < obstacle.x + obstacle.width) and \
                   (obstacle.y < dot.y < obstacle.y + obstacle.height or obstacle.y < dot.y + 10 < obstacle.y + obstacle.height):
                    collision = True

            # Center penalty
            distance_from_center = np.sqrt((dot.x - WIDTH/2)**2 + (dot.y - HEIGHT/2)**2)
            center_penalty = -distance_from_center / 100.0
            agent_instance.store_transition(state, action, center_penalty)

            if len(set(dot.prev_positions)) == 1:
                stationary_penalty = -1500000
                agent_instance.store_transition(state, action, stationary_penalty)

            if collision:
                agent_instance.store_transition(state, action, -1000)
                agent_instance.train()
                running = False
            else:
                agent_instance.store_transition(state, action, 1)

            pygame.display.flip()
            pygame.time.Clock().tick(60)

        scores.append(score)
        print(f"Episode: {episode + 1}, Score: {score}")

    # Plotting scores
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score over Episodes')
    plt.show()

    print(f"Average Score over {NUM_EPISODES} episodes: {sum(scores) / len(scores)}")
    pygame.quit()

if __name__ == "__main__":
    main()

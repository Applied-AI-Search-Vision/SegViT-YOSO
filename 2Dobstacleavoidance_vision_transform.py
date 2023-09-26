import pygame.surfarray as surfarray
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
DOT_SPEED = 7
OBSTACLE_SPEED = 8
NUM_EPISODES = 100

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Vision Transformer components
class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PolicyNetwork(nn.Module):
    def __init__(
        self, in_channels=3, num_classes=4, image_size=800, patch_size=40, embed_size=768, heads=8, layers=12, forward_expansion=4, dropout=0
    ):
        super(PolicyNetwork, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_dim = image_size // patch_size

        self.patch_embedding = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.randn(1, (self.patches_per_dim ** 2) + 1, embed_size))


        self.class_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(layers)]
        )

        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_size)

        class_tokens = self.class_token.expand(N, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)
        x += self.position_embeddings[:, :x.size(1), :]

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, x, x, mask=None)


        x = x.mean(dim=1)
        return torch.nn.functional.softmax(self.fc_out(x), dim=1)

# Game components
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

class ViTAgent(Agent):
    def __init__(self, input_dim=(3, 800, 600), num_classes=4, patch_size=40, lr=0.001):
        super().__init__()
        self.policy_network = PolicyNetwork(in_channels=input_dim[0], num_classes=num_classes, image_size=input_dim[1], patch_size=patch_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def choose_action(self, screen):
        # Convert the screen to tensor format
        state_tensor = torch.FloatTensor(surfarray.array3d(screen)).permute(2, 0, 1).unsqueeze(0)
        
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

    def store_transition(self, screen, action, reward):
        # Store the screen instead of the traditional state
        self.transitions.append((screen, action, reward))

    def train(self):
        if not self.transitions:
            return

        screens, actions, rewards = zip(*self.transitions)
        states = [torch.FloatTensor(surfarray.array3d(screen)).permute(2, 0, 1).unsqueeze(0) for screen in screens]
        states = torch.cat(states, dim=0)
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

        self.transitions = []

# Main game loop
def main_with_vit():
    agent_instance = ViTAgent()
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

            action = agent_instance.choose_action(screen)
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
            agent_instance.store_transition(screen, action, center_penalty)

            if len(set(dot.prev_positions)) == 1:
                stationary_penalty = -80
                agent_instance.store_transition(screen, action, stationary_penalty)

            if collision:
                agent_instance.store_transition(screen, action, -1000)
                agent_instance.train()
                running = False
            else:
                agent_instance.store_transition(screen, action, 1)

            pygame.display.flip()
            pygame.time.Clock().tick(60)

        scores.append(score)
        print(f"Episode: {episode + 1}, Score: {score}")

    # Plotting scores
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score over Episodes with ViT')
    plt.show()

    print(f"Average Score over {NUM_EPISODES} episodes with ViT: {sum(scores) / len(scores)}")
    pygame.quit()

if __name__ == "__main__":
    main_with_vit()

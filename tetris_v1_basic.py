#
# 교육 환경 설정 및 간단한 파이썬 연습 코드
# 기능 : 기본 테트리스 게임 (브랜치: main → develop)
#
# 작성일 : 2026-07-06
# 작성자 : 백정열, SKALA
#
# 변경일 : 
#
# All Rights Reserved by SK AX, SKALA
#

import pygame
import random

# ─────────────────────────────────────────────
# 상수 설정
# ─────────────────────────────────────────────
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
COLS = 10
ROWS = 20

# 색상
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
GRAY   = (128, 128, 128)
COLORS = [
    (0, 240, 240),   # I - 하늘색
    (0, 0, 240),     # J - 파란색
    (240, 160, 0),   # L - 주황색
    (240, 240, 0),   # O - 노란색
    (0, 240, 0),     # S - 초록색
    (160, 0, 240),   # T - 보라색
    (240, 0, 0),     # Z - 빨간색
]

# 테트로미노 모양 (4×4 행렬)
SHAPES = [
    [[1,1,1,1]],                      # I
    [[1,0,0],[1,1,1]],                # J
    [[0,0,1],[1,1,1]],                # L
    [[1,1],[1,1]],                    # O
    [[0,1,1],[1,1,0]],               # S
    [[0,1,0],[1,1,1]],               # T
    [[1,1,0],[0,1,1]],               # Z
]

# ─────────────────────────────────────────────
# 테트로미노 클래스
# ─────────────────────────────────────────────
class Tetromino:
    def __init__(self):
        self.index = random.randint(0, len(SHAPES) - 1)
        self.shape = [row[:] for row in SHAPES[self.index]]
        self.color = COLORS[self.index]
        self.x = COLS // 2 - len(self.shape[0]) // 2
        self.y = 0

    def rotate(self):
        """시계 방향 90도 회전"""
        self.shape = [
            [self.shape[r][c] for r in range(len(self.shape) - 1, -1, -1)]
            for c in range(len(self.shape[0]))
        ]


# ─────────────────────────────────────────────
# 게임 보드
# ─────────────────────────────────────────────
class Board:
    def __init__(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]

    def is_valid(self, piece, dx=0, dy=0, shape=None):
        """충돌 검사"""
        s = shape if shape else piece.shape
        for r, row in enumerate(s):
            for c, cell in enumerate(row):
                if cell:
                    nx, ny = piece.x + c + dx, piece.y + r + dy
                    if nx < 0 or nx >= COLS or ny >= ROWS:
                        return False
                    if ny >= 0 and self.grid[ny][nx]:
                        return False
        return True

    def lock(self, piece):
        """블록을 보드에 고정"""
        for r, row in enumerate(piece.shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[piece.y + r][piece.x + c] = piece.color

    def clear_lines(self):
        """꽉 찬 줄 제거 후 행 수 반환"""
        full = [r for r in range(ROWS) if all(self.grid[r])]
        for r in full:
            del self.grid[r]
            self.grid.insert(0, [None] * COLS)
        return len(full)

    def draw(self, surface):
        """보드 그리기"""
        for r in range(ROWS):
            for c in range(COLS):
                color = self.grid[r][c]
                rect = pygame.Rect(c * BLOCK_SIZE, r * BLOCK_SIZE, BLOCK_SIZE - 1, BLOCK_SIZE - 1)
                if color:
                    pygame.draw.rect(surface, color, rect)
                else:
                    pygame.draw.rect(surface, GRAY, rect, 1)


def draw_piece(surface, piece, offset_x=0, offset_y=0):
    """현재 블록 그리기"""
    for r, row in enumerate(piece.shape):
        for c, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    (piece.x + c + offset_x) * BLOCK_SIZE,
                    (piece.y + r + offset_y) * BLOCK_SIZE,
                    BLOCK_SIZE - 1, BLOCK_SIZE - 1
                )
                pygame.draw.rect(surface, piece.color, rect)


# ─────────────────────────────────────────────
# 메인 게임 루프
# ─────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("테트리스")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    board = Board()
    piece = Tetromino()
    fall_time = 0
    fall_speed = 500  # ms

    running = True
    game_over = False

    while running:
        dt = clock.tick(60)
        screen.fill(BLACK)

        # ── 이벤트 처리 ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if board.is_valid(piece, dx=-1):
                        piece.x -= 1
                elif event.key == pygame.K_RIGHT:
                    if board.is_valid(piece, dx=1):
                        piece.x += 1
                elif event.key == pygame.K_DOWN:
                    if board.is_valid(piece, dy=1):
                        piece.y += 1
                elif event.key == pygame.K_UP:
                    rotated = [
                        [piece.shape[r][c] for r in range(len(piece.shape) - 1, -1, -1)]
                        for c in range(len(piece.shape[0]))
                    ]
                    if board.is_valid(piece, shape=rotated):
                        piece.shape = rotated
                elif event.key == pygame.K_SPACE:
                    while board.is_valid(piece, dy=1):
                        piece.y += 1

        # ── 자동 낙하 ──
        if not game_over:
            fall_time += dt
            if fall_time >= fall_speed:
                fall_time = 0
                if board.is_valid(piece, dy=1):
                    piece.y += 1
                else:
                    board.lock(piece)
                    board.clear_lines()
                    piece = Tetromino()
                    if not board.is_valid(piece):
                        game_over = True

        # ── 그리기 ──
        board.draw(screen)
        if not game_over:
            draw_piece(screen, piece)

        if game_over:
            msg = font.render("GAME OVER", True, WHITE)
            screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, SCREEN_HEIGHT // 2))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
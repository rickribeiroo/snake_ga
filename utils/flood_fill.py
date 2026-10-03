from collections import deque
import numpy as np

try:
    from numba import njit

    @njit(fastmath=True)
    def _numba_bfs(grid, sx, sy, width, height):
        if sx < 0 or sx >= width or sy < 0 or sy >= height:
            return 0.0
        if grid[sy, sx] != 0:
            return 0.0

        visited = np.zeros((height, width), dtype=np.bool_)
        q_x = np.empty(width * height, dtype=np.int32)
        q_y = np.empty(width * height, dtype=np.int32)
        head = 0
        tail = 0

        q_x[tail] = sx
        q_y[tail] = sy
        tail += 1
        visited[sy, sx] = True
        count = 0

        while head < tail:
            cx = q_x[head]
            cy = q_y[head]
            head += 1
            count += 1

            if cy > 0 and not visited[cy - 1, cx] and grid[cy - 1, cx] == 0:
                visited[cy - 1, cx] = True
                q_x[tail] = cx
                q_y[tail] = cy - 1
                tail += 1
            if cx + 1 < width and not visited[cy, cx + 1] and grid[cy, cx + 1] == 0:
                visited[cy, cx + 1] = True
                q_x[tail] = cx + 1
                q_y[tail] = cy
                tail += 1
            if cy + 1 < height and not visited[cy + 1, cx] and grid[cy + 1, cx] == 0:
                visited[cy + 1, cx] = True
                q_x[tail] = cx
                q_y[tail] = cy + 1
                tail += 1
            if cx > 0 and not visited[cy, cx - 1] and grid[cy, cx - 1] == 0:
                visited[cy, cx - 1] = True
                q_x[tail] = cx - 1
                q_y[tail] = cy
                tail += 1

        return count / (width * height)

    _dummy_grid = np.zeros((15, 15), dtype=np.int8)
    _ = _numba_bfs(_dummy_grid, 0, 0, 15, 15)
    NUMBA_AVAILABLE = True

except Exception:
    NUMBA_AVAILABLE = False


def compute_free_space(width, height, snake_body, start_pos, limit=None, grid=None):
    sx, sy = start_pos

    if sx < 0 or sx >= width or sy < 0 or sy >= height:
        return 0.0

    if NUMBA_AVAILABLE and grid is not None:
        return float(_numba_bfs(grid, sx, sy, width, height))

    body_set = set(snake_body) if not isinstance(snake_body, set) else snake_body
    if start_pos in body_set:
        return 0.0

    visited = {start_pos}
    queue = deque([start_pos])
    count = 0
    total_cells = width * height

    while queue:
        cx, cy = queue.popleft()
        count += 1

        if limit is not None and count >= limit:
            return 1.0

        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = (nx, ny)
                if neighbor not in body_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return count / total_cells

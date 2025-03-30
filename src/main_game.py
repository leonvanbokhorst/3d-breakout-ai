import sys

print("Python sys.path:", sys.path)

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# --- Global variables ---
torus_vertices = None
torus_normals = None
torus_indices = None
paddle_vertices = None
paddle_normals = None
paddle_indices = None
paddle_radius = 0.0
paddle_width_deg = 0.0
paddle_height = 0.0
rotation_x = 0.0
rotation_y = 0.0
paddle_angle = 0.0
sphere_vertices = None
sphere_normals = None
sphere_indices = None
ball_radius = 0.1
ball_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ball_vel = np.array([-0.5, -0.3, 0.0], dtype=np.float32)
last_time = 0.0


# --- Geometry Generation ---
def create_torus(major_radius, minor_radius, num_major, num_minor):
    vertices = []
    normals = []
    indices = []

    major_step = 2.0 * np.pi / num_major
    minor_step = 2.0 * np.pi / num_minor

    for i in range(num_major):
        theta = i * major_step
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for j in range(num_minor):
            phi = j * minor_step
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            # Vertex position
            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = (major_radius + minor_radius * cos_phi) * sin_theta
            z = minor_radius * sin_phi
            vertices.extend([x, y, z])

            # Normal vector calculation (points outwards from the tube center)
            nx = cos_phi * cos_theta
            ny = cos_phi * sin_theta
            nz = sin_phi
            # Normalize the normal vector (it should be unit length already, but good practice)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            normals.extend([nx / norm, ny / norm, nz / norm])

    # Generate indices for triangles (connecting vertices to form quads, then splitting quads)
    for i in range(num_major):
        i_next = (i + 1) % num_major
        for j in range(num_minor):
            j_next = (j + 1) % num_minor

            # Indices of the four corners of the current quad
            v0 = i * num_minor + j
            v1 = i_next * num_minor + j
            v2 = i_next * num_minor + j_next
            v3 = i * num_minor + j_next

            # First triangle (v0, v1, v2)
            indices.extend([v0, v1, v2])
            # Second triangle (v0, v2, v3)
            indices.extend([v0, v2, v3])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32),  # Use unsigned int for indices
    )


def create_paddle_segment(paddle_radius, width_degrees, height, segments):
    vertices = []
    normals = []
    indices = []

    half_width_rad = np.radians(width_degrees / 2.0)
    angle_step = np.radians(width_degrees) / segments

    # Generate vertices and normals for the curved paddle surface
    for i in range(segments + 1):
        angle = -half_width_rad + i * angle_step
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Outer edge vertices (top and bottom)
        x_outer = paddle_radius * cos_a
        y_outer = paddle_radius * sin_a
        vertices.extend([x_outer, y_outer, height / 2.0])  # Top outer
        vertices.extend([x_outer, y_outer, -height / 2.0])  # Bottom outer

        # Normals (pointing outwards along the radius)
        normals.extend([cos_a, sin_a, 0.0])  # Normal for top
        normals.extend([cos_a, sin_a, 0.0])  # Normal for bottom

    # Generate indices for triangles
    for i in range(segments):
        idx0 = i * 2  # Top-left
        idx1 = i * 2 + 1  # Bottom-left
        idx2 = (i + 1) * 2  # Top-right
        idx3 = (i + 1) * 2 + 1  # Bottom-right

        # Quad split into two triangles (Corrected Winding Order)
        indices.extend([idx0, idx2, idx1])  # Triangle 1 (Swapped idx1 and idx2)
        indices.extend([idx2, idx3, idx1])  # Triangle 2 (Swapped idx1 and idx3)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


def create_sphere(radius, slices, stacks):
    vertices = []
    normals = []
    indices = []

    # Add top vertex
    vertices.extend([0, 0, radius])
    normals.extend([0, 0, 1])

    # Generate vertices and normals for stacks/slices
    for i in range(1, stacks):
        phi = np.pi * i / stacks
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        for j in range(slices + 1):  # +1 to wrap around
            theta = 2 * np.pi * j / slices
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x = radius * sin_phi * cos_theta
            y = radius * sin_phi * sin_theta
            z = radius * cos_phi
            vertices.extend([x, y, z])

            nx, ny, nz = (
                x / radius,
                y / radius,
                z / radius,
            )  # Normal is just normalized position
            normals.extend([nx, ny, nz])

    # Add bottom vertex
    vertices.extend([0, 0, -radius])
    normals.extend([0, 0, -1])

    # Generate indices
    # Top cap triangles
    for j in range(slices):
        indices.extend([0, j + 1, (j + 1) % slices + 1])

    # Middle stack triangles
    for i in range(stacks - 2):
        stack_start_idx = 1 + i * (slices + 1)
        next_stack_start_idx = 1 + (i + 1) * (slices + 1)
        for j in range(slices):
            v0 = stack_start_idx + j
            v1 = stack_start_idx + j + 1
            v2 = next_stack_start_idx + j + 1
            v3 = next_stack_start_idx + j
            indices.extend([v0, v1, v2])  # Triangle 1
            indices.extend([v0, v2, v3])  # Triangle 2

    # Bottom cap triangles
    bottom_vertex_idx = len(vertices) // 3 - 1
    last_stack_start_idx = 1 + (stacks - 2) * (slices + 1)
    for j in range(slices):
        indices.extend(
            [
                bottom_vertex_idx,
                last_stack_start_idx + (j + 1) % slices,
                last_stack_start_idx + j,
            ]
        )

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


# --- Collision Detection Functions ---


def normalize_angle(angle_rad):
    """Normalize angle to be within -pi to pi."""
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    while angle_rad <= -np.pi:
        angle_rad += 2 * np.pi
    return angle_rad


def check_paddle_collision(
    ball_pos,
    ball_radius,
    paddle_angle_deg,
    paddle_radius,
    paddle_width_deg,
    paddle_height,
):
    """Check for collision between ball and paddle."""
    # print(f"Checking paddle collision: Ball={ball_pos.round(2)}, Paddle Angle={paddle_angle_deg:.1f}") # Optional: General check entry

    # Calculate ball's cylindrical coordinates (radius in XY plane, angle, z)
    ball_dist_xy = np.sqrt(ball_pos[0] ** 2 + ball_pos[1] ** 2)
    ball_angle_rad = np.arctan2(ball_pos[1], ball_pos[0])
    ball_z = ball_pos[2]

    # Paddle parameters in radians
    paddle_angle_rad = np.radians(paddle_angle_deg)
    half_width_rad = np.radians(paddle_width_deg / 2.0)
    # paddle_min_angle = normalize_angle(paddle_angle_rad - half_width_rad)
    # paddle_max_angle = normalize_angle(paddle_angle_rad + half_width_rad)

    # Check Z-height overlap
    z_check_val = paddle_height / 2.0 + ball_radius
    z_overlap = abs(ball_z) < z_check_val
    print(
        f"  Z Check: |{ball_z:.2f}| < {z_check_val:.2f} -> {z_overlap}"
    )  # <--- Uncommented Debug Z
    if not z_overlap:
        return False, None

    # Check radial distance overlap
    radial_check_val = (
        ball_radius  # Check if distance between radii is less than ball radius
    )
    radial_overlap = abs(ball_dist_xy - paddle_radius) < radial_check_val
    print(
        f"  Radial Check: |{ball_dist_xy:.2f} - {paddle_radius:.2f}| < {radial_check_val:.2f} -> {radial_overlap}"
    )  # <--- Uncommented Debug Radial
    if not radial_overlap:
        return False, None

    # Check angular overlap (handling wrap-around)
    delta_angle = normalize_angle(ball_angle_rad - paddle_angle_rad)
    angular_overlap = abs(delta_angle) < half_width_rad
    print(
        f"  Angular Check: Ball Angle={np.degrees(ball_angle_rad):.1f}, Paddle Angle={paddle_angle_deg:.1f}, Delta={np.degrees(delta_angle):.1f}, HalfWidth={np.degrees(half_width_rad):.1f} -> {angular_overlap}"
    )  # Debug Angular

    if angular_overlap:
        # Collision detected! Calculate normal (radially outward)
        print("  !!! PADDLE COLLISION DETECTED !!!")  # Debug Success
        normal = np.array(
            [np.cos(ball_angle_rad), np.sin(ball_angle_rad), 0.0], dtype=np.float32
        )
        return True, normal
    else:
        return False, None


# --- OpenGL Setup ---
def setup_gl(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height) if height > 0 else 1, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Lighting Setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    # Set light position (e.g., directional light from above-right-front)
    glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])  # w=0 for directional
    # Set light color properties
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])  # Low ambient
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])  # White diffuse
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])  # White specular

    glEnable(GL_COLOR_MATERIAL)  # Allow setting material color via glColor
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    # Add some basic material shininess for specular highlights
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

    glShadeModel(GL_SMOOTH)  # Enable smooth shading
    glEnable(GL_CULL_FACE)  # <--- Re-enable face culling
    glCullFace(GL_BACK)  # <--- Re-enable back face culling


# --- Key Callback (for paddle movement) ---
def key_callback(window, key, scancode, action, mods):
    global paddle_angle
    paddle_speed = 5.0  # Degrees per key press
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_RIGHT:
            paddle_angle -= paddle_speed
        elif key == glfw.KEY_LEFT:
            paddle_angle += paddle_speed
        # Keep angle between 0 and 360
        paddle_angle %= 360


# --- Main Function ---
def main():
    global torus_vertices, torus_normals, torus_indices, rotation_x, rotation_y
    global paddle_vertices, paddle_normals, paddle_indices, paddle_angle
    global paddle_radius, paddle_width_deg, paddle_height
    global sphere_vertices, sphere_normals, sphere_indices, ball_pos, ball_vel, last_time

    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        sys.exit()

    # Create window
    window = glfw.create_window(800, 600, "3D Breakout - Doughnut Arena!", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    print("OpenGL Version:", glGetString(GL_VERSION).decode())
    print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
    print("Renderer:", glGetString(GL_RENDERER).decode())

    # --- Set Keyboard Callback ---
    glfw.set_key_callback(window, key_callback)

    # --- Generate Torus Geometry ---
    major_radius = 1.5
    minor_radius = 0.5
    num_major = 40
    num_minor = 30
    torus_vertices, torus_normals, torus_indices = create_torus(
        major_radius, minor_radius, num_major, num_minor
    )

    # --- Generate Paddle Geometry ---
    paddle_radius = major_radius + minor_radius + 0.1
    paddle_width_deg = 30.0
    paddle_height = 0.2
    paddle_segments = 10
    paddle_vertices, paddle_normals, paddle_indices = create_paddle_segment(
        paddle_radius, paddle_width_deg, paddle_height, paddle_segments
    )

    # --- Generate Sphere Geometry ---
    slices, stacks = 16, 8
    sphere_vertices, sphere_normals, sphere_indices = create_sphere(
        ball_radius, slices, stacks
    )

    glClearColor(0.1, 0.1, 0.2, 1.0)
    last_time = glfw.get_time()

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # --- Time Calculation ---
        current_time = glfw.get_time()
        dt = current_time - last_time
        last_time = current_time

        width, height = glfw.get_framebuffer_size(window)
        setup_gl(width, height)

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --- Common Camera Position ---
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)

        # --- Update Ball Position ---
        ball_pos += ball_vel * dt
        print(
            f"Frame {int(current_time*10)}: dt={dt:.4f}, ball_pos={ball_pos.round(2)}, ball_vel={ball_vel.round(2)}"
        )

        paddle_hit = False  # Flag to see if paddle collision happened
        # --- Check Paddle Collision FIRST ---
        collided, normal = check_paddle_collision(
            ball_pos,
            ball_radius,
            paddle_angle,
            paddle_radius,
            paddle_width_deg,
            paddle_height,
        )
        if collided:
            paddle_hit = True
            vel_dot_normal = np.dot(ball_vel, normal)
            # Only bounce if moving towards the paddle
            if vel_dot_normal < 0:
                reflected_vel = ball_vel - 2 * vel_dot_normal * normal
                print(
                    f"Paddle Bounce! Normal: {normal.round(2)}, Old Vel: {ball_vel.round(2)}, New Vel: {reflected_vel.round(2)}"
                )
                ball_vel = reflected_vel
                # Nudge ball slightly away
                nudge_factor = 0.01
                ball_pos += normal * nudge_factor

        # --- Check Torus Collision (only if paddle didn't hit) ---
        if not paddle_hit:
            dist_xy_sq = ball_pos[0] ** 2 + ball_pos[1] ** 2
            if dist_xy_sq > 1e-6:
                dist_xy = np.sqrt(dist_xy_sq)
                closest_center_x = ball_pos[0] / dist_xy * major_radius
                closest_center_y = ball_pos[1] / dist_xy * major_radius
                dist_to_centerline_sq = (
                    (ball_pos[0] - closest_center_x) ** 2
                    + (ball_pos[1] - closest_center_y) ** 2
                    + ball_pos[2] ** 2
                )

                if dist_to_centerline_sq < (minor_radius + ball_radius) ** 2:
                    # Collision detected!

                    # --- Refine collision point estimation ---
                    # Vector from centerline point to ball pos
                    vec_to_ball = ball_pos - np.array(
                        [closest_center_x, closest_center_y, 0.0]
                    )
                    vec_to_ball_len = np.linalg.norm(vec_to_ball)
                    if vec_to_ball_len < 1e-6:
                        normal = np.array(
                            [ball_pos[0] / dist_xy, ball_pos[1] / dist_xy, 0.0]
                        )
                    else:
                        dir_to_ball = vec_to_ball / vec_to_ball_len
                        # Estimate the actual point on the torus surface
                        surface_point = (
                            np.array([closest_center_x, closest_center_y, 0.0])
                            + dir_to_ball * minor_radius
                        )
                        coll_theta = np.arctan2(surface_point[1], surface_point[0])
                        coll_phi = np.arctan2(
                            dir_to_ball[2],
                            np.sqrt(dir_to_ball[0] ** 2 + dir_to_ball[1] ** 2),
                        )
                        cos_theta = np.cos(coll_theta)
                        sin_theta = np.sin(coll_theta)
                        cos_phi = np.cos(coll_phi)
                        sin_phi = np.sin(coll_phi)
                        nx = cos_phi * cos_theta
                        ny = cos_phi * sin_theta
                        nz = sin_phi
                        normal = np.array([nx, ny, nz])
                        # Ensure normal is unit length
                        norm_len_check = np.linalg.norm(normal)
                        if norm_len_check > 1e-6:
                            normal /= norm_len_check
                        else:
                            # Fallback if normal calculation still fails (should be rare now)
                            normal = (
                                dir_to_ball  # Use direction from center as fallback
                            )

                    # 3. Calculate reflection R = V - 2 * dot(V, N) * N
                    vel_dot_normal = np.dot(ball_vel, normal)
                    # Only bounce if moving towards the torus surface roughly
                    if vel_dot_normal < 0:
                        reflected_vel = ball_vel - 2 * vel_dot_normal * normal
                        print(
                            f"Torus Bounce! Angles: ({np.degrees(coll_theta):.1f}, {np.degrees(coll_phi):.1f}), "
                            f"Normal: {normal.round(2)}, Old Vel: {ball_vel.round(2)}, New Vel: {reflected_vel.round(2)}"
                        )
                        ball_vel = reflected_vel
                        nudge_factor = 0.01
                        ball_pos += normal * nudge_factor

        # --- Drawing the Torus ---
        glPushMatrix()
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        glColor3f(1.0, 0.7, 0.1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, torus_vertices)
        glNormalPointer(GL_FLOAT, 0, torus_normals)

        glDrawElements(GL_TRIANGLES, len(torus_indices), GL_UNSIGNED_INT, torus_indices)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glPopMatrix()

        # --- Drawing the Paddle ---
        glPushMatrix()
        glRotatef(paddle_angle, 0, 0, 1)

        glColor3f(0.2, 0.8, 0.3)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, paddle_vertices)
        glNormalPointer(GL_FLOAT, 0, paddle_normals)

        glDrawElements(
            GL_TRIANGLES, len(paddle_indices), GL_UNSIGNED_INT, paddle_indices
        )

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glPopMatrix()

        # --- Drawing the Ball ---
        glPushMatrix()
        glTranslatef(ball_pos[0], ball_pos[1], ball_pos[2])

        glColor3f(0.9, 0.9, 0.9)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, sphere_vertices)
        glNormalPointer(GL_FLOAT, 0, sphere_normals)

        glDrawElements(
            GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, sphere_indices
        )

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glPopMatrix()

        # --- Update Torus Rotation (Optional - can be distracting) ---
        # rotation_x += 0.3
        # rotation_y += 0.2

        # Swap buffers
        glfw.swap_buffers(window)

        # Poll events
        glfw.poll_events()

    # Terminate
    glfw.terminate()
    sys.exit()


if __name__ == "__main__":
    main()

#include "common.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
// Put any static global variables here that you will use throughout the simulation.
double x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound;
int rank_x, rank_y, grid_size;

std::vector<particle_t> ghost_to_left;
std::vector<particle_t> ghost_to_right;
std::vector<particle_t> ghost_to_above;
std::vector<particle_t> ghost_to_below;
std::vector<particle_t> ghost_to_top_left;
std::vector<particle_t> ghost_to_top_right;
std::vector<particle_t> ghost_to_bottom_left;
std::vector<particle_t> ghost_to_bottom_right;

std::vector<particle_t> ghost_from_left;
std::vector<particle_t> ghost_from_right;
std::vector<particle_t> ghost_from_above;
std::vector<particle_t> ghost_from_below;
std::vector<particle_t> ghost_from_top_left;
std::vector<particle_t> ghost_from_top_right;
std::vector<particle_t> ghost_from_bottom_left;
std::vector<particle_t> ghost_from_bottom_right;

// Apply force between two particles
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void apply_force(particle_t& particle, double& neighbor_x, double& neighbor_y) {
    // Calculate Distance
    double dx = neighbor_x - particle.x;
    double dy = neighbor_y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);  // Ensure minimum distance between particles
    double r = sqrt(r2);  // Compute distance between particles

    // Short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    // Boundary conditions
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

double lower_bound;
double upper_bound;

std::vector<particle_t> local_parts;

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    grid_size = static_cast<int>(sqrt(num_procs));
    if (grid_size * grid_size != num_procs) {
        std::cerr << "Error: Number of processes must be a perfect square!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    } // square shape grid

    double cell_width = size / grid_size;
    double cell_height = size / grid_size;

    rank_x = rank % grid_size;
    rank_y = rank % grid_size;

    x_lower_bound = rank_x * cell_width;
    x_upper_bound = (rank_x + 1) * cell_width;
    y_lower_bound = rank_y * cell_height;
    y_upper_bound = (rank_y + 1) * cell_height;
    
    // Store local particles relevant to this rank
    local_parts.clear(); // initialize 
    
    // If particle is within the bounds of the rank, add it to local_parts
    for (int i = 0; i < num_parts; i++) {
        if (parts[i].x >= x_lower_bound && parts[i].x < x_upper_bound && 
            parts[i].y >= y_lower_bound && parts[i].y < y_upper_bound) {
            local_parts.push_back(parts[i]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Initialization 
    ghost_to_left.clear();
    ghost_to_right.clear();
    ghost_to_above.clear();
    ghost_to_below.clear();
    ghost_to_top_left.clear();
    ghost_to_top_right.clear();
    ghost_to_bottom_left.clear();
    ghost_to_bottom_right.clear();

    ghost_from_left.clear();
    ghost_from_right.clear();
    ghost_from_above.clear();
    ghost_from_below.clear();
    ghost_from_top_left.clear();
    ghost_from_top_right.clear();
    ghost_from_bottom_left.clear();
    ghost_from_bottom_right.clear();

    // ============================== MOVE PARTICLES ================================= //

    // Define rank horizontal and vertical
    int rank_left = (rank_x > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_right = (rank_x < grid_size - 1) ? rank + 1 : MPI_PROC_NULL;
    int rank_above = (rank_y > 0) ? rank - grid_size : MPI_PROC_NULL;
    int rank_below = (rank_y < grid_size - 1) ? rank + grid_size : MPI_PROC_NULL;
    
    int rank_top_left = (rank_x > 0 && rank_y > 0) ? rank - grid_size - 1 : MPI_PROC_NULL;
    int rank_top_right = (rank_x < grid_size - 1 && rank_y > 0) ? rank - grid_size + 1 : MPI_PROC_NULL;
    int rank_bottom_left = (rank_x > 0 && rank_y < grid_size - 1) ? rank + grid_size - 1 : MPI_PROC_NULL;
    int rank_bottom_right = (rank_x < grid_size - 1 && rank_y < grid_size - 1) ? rank + grid_size + 1 : MPI_PROC_NULL;

    

    std::cout << "Rank " << rank << " neighbors: "
          << "left=" << rank_left << ", "
          << "right=" << rank_right << ", "
          << "above=" << rank_above << ", "
          << "below=" << rank_below << std::endl;
    

    // Iterate over local particles to determine ghost particle counts
    for (size_t i = 0; i < local_parts.size(); i++) {
        particle_t& p = local_parts[i];

        // horizontal side
        if (p.x - x_lower_bound < cutoff && 
            p.y >= y_lower_bound && p.y < y_upper_bound) {
            ghost_to_left.push_back(p);
        }
        if (x_upper_bound - p.x < cutoff &&
            p.y >= y_lower_bound && p.y < y_upper_bound) {
            ghost_to_right.push_back(p);
        }
        // vertical side
        if (p.y - y_lower_bound < cutoff &&
            p.x >= x_lower_bound && p.x < x_upper_bound) {
            ghost_to_below.push_back(p);
        }
        if (y_upper_bound - p.y < cutoff &&
            p.x >= x_lower_bound && p.x < x_upper_bound) {
            ghost_to_above.push_back(p);
        }
        // diagonal side
        if (p.x - x_lower_bound < cutoff && 
            p.y - y_lower_bound < cutoff) {
            ghost_to_bottom_left.push_back(p);
        }
        if (x_upper_bound - p.x < cutoff &&
            p.y - y_lower_bound < cutoff) {
            ghost_to_bottom_right.push_back(p);
        }
        if (p.x - x_lower_bound < cutoff &&
            y_upper_bound - p.y < cutoff) {
            ghost_to_top_left.push_back(p);
        }
        if (x_upper_bound - p.x < cutoff && 
            y_upper_bound - p.y < cutoff) {
            ghost_to_top_right.push_back(p);
        }
    }

    // Set ghost counts based on vector sizes
    int ghost_to_above_count = ghost_to_above.size();
    int ghost_to_below_count = ghost_to_below.size();
    int ghost_to_left_count = ghost_to_left.size();
    int ghost_to_right_count = ghost_to_right.size();
    int ghost_to_top_left_count = ghost_to_top_left.size();
    int ghost_to_top_right_count = ghost_to_top_right.size();
    int ghost_to_bottom_left_count = ghost_to_bottom_left.size();   
    int ghost_to_bottom_right_count = ghost_to_bottom_right.size();

    int ghost_from_above_count = 0;
    int ghost_from_below_count = 0;
    int ghost_from_left_count = 0;
    int ghost_from_right_count = 0;
    int ghost_from_top_left_count = 0;
    int ghost_from_top_right_count = 0;
    int ghost_from_bottom_left_count = 0;
    int ghost_from_bottom_right_count = 0;

    // ============================== SEND / RECEIVE GHOST PARTICLE COUNTS ================================= //
    // Define particle_requests array and initialize it
    MPI_Request requests[32]; // 이름을 통일
    for (int i = 0; i < 32; i++) {
        requests[i] = MPI_REQUEST_NULL; // 올바르게 초기화
    }
    
    int req_index = 0;

    // Non-blocking send and receive for ghost counts
    // Horizontal
    // 수신 요청 (Irecv 먼저 설정)
    if (rank_left >= 0) {
        MPI_Irecv(&ghost_from_left_count, 1, MPI_INT, rank_left, 0, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_right >= 0) {
        MPI_Irecv(&ghost_from_right_count, 1, MPI_INT, rank_right, 1, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_above >= 0) {
        MPI_Irecv(&ghost_from_above_count, 1, MPI_INT, rank_above, 2, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_below >= 0) {
        MPI_Irecv(&ghost_from_below_count, 1, MPI_INT, rank_below, 3, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_top_left >= 0) {
        MPI_Irecv(&ghost_from_top_left_count, 1, MPI_INT, rank_top_left, 4, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_top_right >= 0) {
        MPI_Irecv(&ghost_from_top_right_count, 1, MPI_INT, rank_top_right, 5, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_bottom_left >= 0) {
        MPI_Irecv(&ghost_from_bottom_left_count, 1, MPI_INT, rank_bottom_left, 6, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_bottom_right >= 0) {
        MPI_Irecv(&ghost_from_bottom_right_count, 1, MPI_INT, rank_bottom_right, 7, MPI_COMM_WORLD, &requests[req_index++]);
    }

    // 송신 요청 (Isend)
    if (rank_left >= 0) {
        MPI_Isend(&ghost_to_left_count, 1, MPI_INT, rank_left, 0, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_right >= 0) {
        MPI_Isend(&ghost_to_right_count, 1, MPI_INT, rank_right, 1, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_above >= 0) {
        MPI_Isend(&ghost_to_above_count, 1, MPI_INT, rank_above, 2, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_below >= 0) {
        MPI_Isend(&ghost_to_below_count, 1, MPI_INT, rank_below, 3, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_top_left >= 0) {
        MPI_Isend(&ghost_to_top_left_count, 1, MPI_INT, rank_top_left, 4, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_top_right >= 0) {
        MPI_Isend(&ghost_to_top_right_count, 1, MPI_INT, rank_top_right, 5, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_bottom_left >= 0) {
        MPI_Isend(&ghost_to_bottom_left_count, 1, MPI_INT, rank_bottom_left, 6, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (rank_bottom_right >= 0) {
        MPI_Isend(&ghost_to_bottom_right_count, 1, MPI_INT, rank_bottom_right, 7, MPI_COMM_WORLD, &requests[req_index++]);
    }

    // Wait for all sends/receives to complete
    MPI_Waitall(req_index, requests, MPI_STATUSES_IGNORE);

    // Resize vectors based on received counts
    ghost_from_left.resize(ghost_from_left_count);
    ghost_from_right.resize(ghost_from_right_count);
    ghost_from_above.resize(ghost_from_above_count);
    ghost_from_below.resize(ghost_from_below_count);
    ghost_from_top_left.resize(ghost_from_top_left_count);
    ghost_from_top_right.resize(ghost_from_top_right_count);
    ghost_from_bottom_left.resize(ghost_from_bottom_left_count);
    ghost_from_bottom_right.resize(ghost_from_bottom_right_count);


    // ============================== SEND / RECEIVE GHOST PARTICLE DATA ================================= //
    // Horizontal (Left / Right)
    if (ghost_from_left_count > 0 && rank_left >= 0) {
        MPI_Irecv(ghost_from_left.data(), ghost_from_left_count, PARTICLE, rank_left, 0, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_right_count > 0 && rank_right >= 0) {
        MPI_Irecv(ghost_from_right.data(), ghost_from_right_count, PARTICLE, rank_right, 1, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_above_count > 0 && rank_above >= 0) {
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_count, PARTICLE, rank_above, 2, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_below_count > 0 && rank_below >= 0) {
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_count, PARTICLE, rank_below, 3, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_top_left_count > 0 && rank_top_left >= 0) {
        MPI_Irecv(ghost_from_top_left.data(), ghost_from_top_left_count, PARTICLE, rank_top_left, 4, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_top_right_count > 0 && rank_top_right >= 0) {
        MPI_Irecv(ghost_from_top_right.data(), ghost_from_top_right_count, PARTICLE, rank_top_right, 5, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_bottom_left_count > 0 && rank_bottom_left >= 0) {
        MPI_Irecv(ghost_from_bottom_left.data(), ghost_from_bottom_left_count, PARTICLE, rank_bottom_left, 6, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_from_bottom_right_count > 0 && rank_bottom_right >= 0) {
        MPI_Irecv(ghost_from_bottom_right.data(), ghost_from_bottom_right_count, PARTICLE, rank_bottom_right, 7, MPI_COMM_WORLD, &requests[req_index++]);
    }
    // 송신 요청 (Isend)
    if (ghost_to_left_count > 0 && rank_left >= 0) {
        MPI_Isend(ghost_to_left.data(), ghost_to_left_count, PARTICLE, rank_left, 0, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_right_count > 0 && rank_right >= 0) {
        MPI_Isend(ghost_to_right.data(), ghost_to_right_count, PARTICLE, rank_right, 1, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_above_count > 0 && rank_above >= 0) {
        MPI_Isend(ghost_to_above.data(), ghost_to_above_count, PARTICLE, rank_above, 2, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_below_count > 0 && rank_below >= 0) {
        MPI_Isend(ghost_to_below.data(), ghost_to_below_count, PARTICLE, rank_below, 3, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_top_left_count > 0 && rank_top_left >= 0) {
        MPI_Isend(ghost_to_top_left.data(), ghost_to_top_left_count, PARTICLE, rank_top_left, 4, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_top_right_count > 0 && rank_top_right >= 0) {
        MPI_Isend(ghost_to_top_right.data(), ghost_to_top_right_count, PARTICLE, rank_top_right, 5, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_bottom_left_count > 0 && rank_bottom_left >= 0) {
        MPI_Isend(ghost_to_bottom_left.data(), ghost_to_bottom_left_count, PARTICLE, rank_bottom_left, 6, MPI_COMM_WORLD, &requests[req_index++]);
    }
    if (ghost_to_bottom_right_count > 0 && rank_bottom_right >= 0) {
        MPI_Isend(ghost_to_bottom_right.data(), ghost_to_bottom_right_count, PARTICLE, rank_bottom_right, 7, MPI_COMM_WORLD, &requests[req_index++]);
    }
    
    

    // Wait for all sends/receives to complete
    MPI_Waitall(req_index, requests, MPI_STATUSES_IGNORE);

        // ============================= Compute Forces ============================= //
    for (int i = 0; i < local_parts.size(); ++i) {
        local_parts[i].ax = local_parts[i].ay = 0;

        for (int j = 0; j < local_parts.size(); ++j) {
            if (i != j) {apply_force(local_parts[i], local_parts[j]);}
        }
        // Compute forces with ghost particles from
        // Above
        for (size_t jj = 0; jj < ghost_from_above.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_above[jj]);
        }
        // Below
        for (size_t jj = 0; jj < ghost_from_below.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_below[jj]);
        }
        // Left
        for (size_t jj = 0; jj < ghost_from_left.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_left[jj]);
        }
        // Right
        for (size_t jj = 0; jj < ghost_from_right.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_right[jj]);
        }
        // Top Left
        for (size_t jj = 0; jj < ghost_from_top_left.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_top_left[jj]);
        }
        // Top Right
        for (size_t jj = 0; jj < ghost_from_top_right.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_top_right[jj]);
        }
        // Bottom Left
        for (size_t jj = 0; jj < ghost_from_bottom_left.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_bottom_left[jj]);
        }
        // Bottom Right
        for (size_t jj = 0; jj < ghost_from_bottom_right.size(); jj += 1) {
            apply_force(local_parts[i], ghost_from_bottom_right[jj]);
        }

    // ============================== MOVE PARTICLES ============================== //
    for (size_t i = 0; i < local_parts.size(); i++) {
        move(local_parts[i], size);
    }
// ============================== PARTICLE EXCHANGE ACROSS RANKS ================================= //

// Iterate over local particles to identify which need to be sent
for (size_t i = 0; i < local_parts.size(); ) {
    particle_t& p = local_parts[i];

    // to above
    if (p.y >= y_upper_bound) {
        ghost_to_above.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to below
    if (p.y < y_lower_bound) {
        ghost_to_below.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to the left
    if (p.x < x_lower_bound) {
        ghost_to_left.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to the right
    if (p.x >= x_upper_bound) {
        ghost_to_right.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }

    // to top left
    if (p.x < x_lower_bound && p.y >= y_upper_bound) {
        ghost_to_top_left.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to top right
    if (p.x >= x_upper_bound && p.y >= y_upper_bound) {
        ghost_to_top_right.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to bottom left
    if (p.x < x_lower_bound && p.y < y_lower_bound) {
        ghost_to_bottom_left.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }
    // to bottom right
    if (p.x >= x_upper_bound && p.y < y_lower_bound) {
        ghost_to_bottom_right.push_back(p);
        local_parts.erase(local_parts.begin() + i);
        continue;
    }

    // Increment index if no erase occurred
    ++i;
}

// ============================== INSERT RECEIVED PARTICLES ================================= //
// Horizontal
local_parts.insert(local_parts.end(), ghost_from_left.begin(), ghost_from_left.end());
local_parts.insert(local_parts.end(), ghost_from_right.begin(), ghost_from_right.end());

// Vertical
local_parts.insert(local_parts.end(), ghost_from_above.begin(),ghost_from_above.end());
local_parts.insert(local_parts.end(), ghost_from_below.begin(), ghost_from_below.end());

// Diagonal
local_parts.insert(local_parts.end(), ghost_from_top_left.begin(), ghost_from_top_left.end());
local_parts.insert(local_parts.end(), ghost_from_top_right.begin(), ghost_from_top_right.end());
local_parts.insert(local_parts.end(), ghost_from_bottom_left.begin(), ghost_from_bottom_left.end());
local_parts.insert(local_parts.end(), ghost_from_bottom_right.begin(), ghost_from_bottom_right.end());


MPI_Barrier(MPI_COMM_WORLD);
}

}


void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Get the local number of particles on each rank
    int local_count = local_parts.size();  // Use local_parts.size() instead of num_parts
    std::vector<int> all_counts(num_procs);

    // Gather particle counts from all ranks
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacement array for MPI_Gatherv
    std::vector<int> displacements(num_procs, 0);

    if (rank == 0) {
        int offset = 0;
        for (int i = 0; i < num_procs; i++) {
            displacements[i] = offset;
            offset += all_counts[i];
        }
    }

    // Rank 0 already has enough space allocated in `parts`, so no need to resize
    MPI_Gatherv(local_parts.data(), local_count, PARTICLE,
                parts, all_counts.data(), displacements.data(),
                PARTICLE, 0, MPI_COMM_WORLD);

    // Rank 0 sorts the gathered particles by ID
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });

        // Print the x-coordinate of the first particle
        if (num_parts > 0) {
            std::cout << "x-coordinate of first particle: "
                      << parts[0].x << std::endl;
        }
    }
}

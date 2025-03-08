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
int flag = 0;


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

    // ============================== MOVE PARTICLES ================================= //

    // Define rank horizontal and vertical
    // maximum rank
    int max_rank = grid_size * grid_size - 1; // 최대 랭크는 0부터 시작하므로 -1 필요

    // Horizontal and vertical neighbors
    int rank_left = (rank_x > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_right = (rank_x < grid_size - 1) ? rank + 1 : MPI_PROC_NULL;
    
    int rank_above = (rank_y > 0 && rank - grid_size >= 0) 
                     ? rank - grid_size : MPI_PROC_NULL;
    
    int rank_below = (rank_y < grid_size - 1 && rank + grid_size <= max_rank) 
                     ? rank + grid_size : MPI_PROC_NULL;
    
    // Diagonal neighbors with strict boundary checking
    int rank_top_left = (rank_x > 0 && rank_y > 0 && rank - grid_size - 1 >= 0) 
                        ? rank - grid_size - 1 : MPI_PROC_NULL;
    
    int rank_top_right = (rank_x < grid_size - 1 && rank_y > 0 && rank - grid_size + 1 >= 0) 
                         ? rank - grid_size + 1 : MPI_PROC_NULL;
    
    int rank_bottom_left = (rank_x > 0 && rank_y < grid_size - 1 && rank + grid_size - 1 <= max_rank) 
                           ? rank + grid_size - 1 : MPI_PROC_NULL;
    
    int rank_bottom_right = (rank_x < grid_size - 1 && rank_y < grid_size - 1 && rank + grid_size + 1 <= max_rank) 
                            ? rank + grid_size + 1 : MPI_PROC_NULL;
    
    // Additional safety check for out-of-bound ranks
    if (rank_left < 0 || rank_left > max_rank) rank_left = MPI_PROC_NULL;
    if (rank_right < 0 || rank_right > max_rank) rank_right = MPI_PROC_NULL;
    if (rank_above < 0 || rank_above > max_rank) rank_above = MPI_PROC_NULL;
    if (rank_below < 0 || rank_below > max_rank) rank_below = MPI_PROC_NULL;
    
    if (rank_top_left < 0 || rank_top_left > max_rank) rank_top_left = MPI_PROC_NULL;
    if (rank_top_right < 0 || rank_top_right > max_rank) rank_top_right = MPI_PROC_NULL;
    if (rank_bottom_left < 0 || rank_bottom_left > max_rank) rank_bottom_left = MPI_PROC_NULL;
    if (rank_bottom_right < 0 || rank_bottom_right > max_rank) rank_bottom_right = MPI_PROC_NULL;
    
    /* std::cout << "Rank " << rank << " neighbors: "
          << "left=" << rank_left << ", "
          << "right=" << rank_right << ", "
          << "above=" << rank_above << ", "
          << "below=" << rank_below << ", "
          << "top_left=" << rank_top_left << ", "
        << "top_right=" << rank_top_right << ", "
        << "bottom_left=" << rank_bottom_left << ", "
        << "bottom_right=" << rank_bottom_right << std::endl; */
    
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
    MPI_Request recv_requests[16];
    MPI_Request send_requests[16];

    // Receive Requests 초기화
    for (int i = 0; i < 16; i++) {
        recv_requests[i] = MPI_REQUEST_NULL;
        send_requests[i] = MPI_REQUEST_NULL;
    }
    // Non-blocking send and receive for ghost counts
    // Horizontal
    // Irecv should be called before Isend
    // std::cout << "Rank " << rank << " before all Irecv calls of particle counts" << std::endl;

    // Receive

    if (rank_left != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_left_count, 1, MPI_INT,
                  rank_left, 0, MPI_COMM_WORLD, &recv_requests[0]);
    }
    if (rank_right != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_right_count, 1, MPI_INT,
                  rank_right, 1, MPI_COMM_WORLD, &recv_requests[1]);
    } 

    if (rank_above != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_above_count, 1, MPI_INT,
                  rank_above, 2, MPI_COMM_WORLD, &recv_requests[2]);
    }
    
    if (rank_below != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_below_count, 1, MPI_INT,
                  rank_below, 3, MPI_COMM_WORLD, &recv_requests[3]);
    } 
    if (rank_top_left != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_top_left_count, 1, MPI_INT,
                  rank_top_left, 4, MPI_COMM_WORLD, &recv_requests[4]);
    }
    if (rank_top_right != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_top_right_count, 1, MPI_INT,
                  rank_top_right, 5, MPI_COMM_WORLD, &recv_requests[5]);
    } 
    if (rank_bottom_left != MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_bottom_left_count, 1, MPI_INT,
                  rank_bottom_left, 6, MPI_COMM_WORLD, &recv_requests[6]);
    } 
    if (rank_bottom_right!= MPI_PROC_NULL) {
        MPI_Irecv(&ghost_from_bottom_right_count, 1, MPI_INT,
                  rank_bottom_right, 7, MPI_COMM_WORLD, &recv_requests[7]);
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " after receiving particle counts" << std::endl;

    if (rank_left != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_left_count, 1, MPI_INT,
                rank_left, 0, MPI_COMM_WORLD, &send_requests[0]);
    } 
    if (rank_right != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_right_count, 1, MPI_INT,
                rank_right, 1, MPI_COMM_WORLD, &send_requests[1]);
    }

    if (rank_above != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_above_count, 1, MPI_INT,
            rank_above, 2, MPI_COMM_WORLD, &send_requests[2]);
    } 

    if (rank_below != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_below_count, 1, MPI_INT,
            rank_below, 3, MPI_COMM_WORLD, &send_requests[3]);
    } 
    if (rank_top_left != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_top_left_count, 1, MPI_INT,
            rank_top_left, 4, MPI_COMM_WORLD, &send_requests[4]);
    } 
    if (rank_top_right != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_top_right_count, 1, MPI_INT,
            rank_top_right, 5, MPI_COMM_WORLD, &send_requests[5]);
    } 
    if (rank_bottom_left != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_bottom_left_count, 1, MPI_INT,
            rank_bottom_left, 6, MPI_COMM_WORLD, &send_requests[6]);
    } 
    if (rank_bottom_right != MPI_PROC_NULL) {
        MPI_Isend(&ghost_to_bottom_right_count, 1, MPI_INT,
            rank_bottom_right, 7, MPI_COMM_WORLD, &send_requests[7]);
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " after sending particle counts" << std::endl;

    // MPI_Waitall(8, recv_requests, MPI_STATUSES_IGNORE);
    // MPI_Waitall(8, send_requests, MPI_STATUSES_IGNORE);

    std:: cout << " count done " << std:: endl;
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
    // To maintain the symetry of send and receive, we need to call and MPI_Irecv even though
    // ghost_to_*_count is zero. 


    // 수신 요청 배열 초기화
    for (int i = 0; i < 16; i++) {
        recv_requests[i] = MPI_REQUEST_NULL;
        send_requests[i] = MPI_REQUEST_NULL;
    }

    // Horizontal (Left / Right)
    if (rank_left != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_left.data(), ghost_from_left_count, PARTICLE,
                  rank_left, 8, MPI_COMM_WORLD, &recv_requests[0]);      
    } 
    if (rank_right != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_right.data(), ghost_from_right_count, PARTICLE,
                  rank_right, 9, MPI_COMM_WORLD, &recv_requests[1]);
                  // std::cout << "right_tag " << create_tag(rank_right, rank, RIGHT) << " recv from right" << std::endl;
    } 
    if (rank_above != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_above.data(), ghost_from_above_count, PARTICLE,
                  rank_above, 10, MPI_COMM_WORLD, &recv_requests[2]);
                  // std :: cout << "above_tag " << create_tag(rank_above, rank, ABOVE) << " recv from above" << std::endl;
    } 
    
    if (rank_below != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_below.data(), ghost_from_below_count, PARTICLE,
                  rank_below, 11, MPI_COMM_WORLD, &recv_requests[3]);
    }
    
    if (rank_top_left != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_top_left.data(), ghost_from_top_left_count, PARTICLE,
                  rank_top_left, 12, MPI_COMM_WORLD, &recv_requests[4]);
    } 
    
    if (rank_top_right != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_top_right.data(), ghost_from_top_right_count, PARTICLE,
                  rank_top_right, 13, MPI_COMM_WORLD, &recv_requests[5]);
    } 
    
    if (rank_bottom_left != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_bottom_left.data(), ghost_from_bottom_left_count, PARTICLE,
                  rank_bottom_left, 14, MPI_COMM_WORLD, &recv_requests[6]);
    }
    
    if (rank_bottom_right != MPI_PROC_NULL) {
        MPI_Irecv(ghost_from_bottom_right.data(), ghost_from_bottom_right_count, PARTICLE,
                  rank_bottom_right, 15, MPI_COMM_WORLD, &recv_requests[7]);
    } 
    
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " after receiving particle data" << std::endl;


    // std::cout << "Rank " << rank << " After all Irecv calls of particle data" << std::endl;
    
    // std::cout << "Rank " << rank << " before all Isend calls of particle data" << std::endl;

    // MPI_Isend
    if (rank_left != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_left.data(), ghost_to_left_count, PARTICLE,
                  rank_left, 8, MPI_COMM_WORLD, &send_requests[0]);
                  // std::cout << "left_tag " << create_tag(rank, rank_left, LEFT) << " send to left" << std::endl;
    }
    
    if (rank_right != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_right.data(), ghost_to_right_count, PARTICLE,
                  rank_right, 9, MPI_COMM_WORLD, &send_requests[1]);
                  // std::cout << "right_tag " << create_tag(rank, rank_right, RIGHT) << " send to right" << std::endl;
    } 
    
    if (rank_above != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_above.data(), ghost_to_above_count, PARTICLE,
                  rank_above, 10, MPI_COMM_WORLD, &send_requests[2]);
                  // std::cout << "above_tag " << create_tag(rank, rank_above, ABOVE) << " send to above" << std::endl;
    } 
    
    if (rank_below != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_below.data(), ghost_to_below_count, PARTICLE,
                  rank_below, 11, MPI_COMM_WORLD, &send_requests[3]);
    } 
    
    if (rank_top_left != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_top_left.data(), ghost_to_top_left_count, PARTICLE,
                  rank_top_left, 12, MPI_COMM_WORLD, &send_requests[4]);
    }
    
    if (rank_top_right != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_top_right.data(), ghost_to_top_right_count, PARTICLE,
                  rank_top_right, 13, MPI_COMM_WORLD, &send_requests[5]);
    } 
    
    if (rank_bottom_left != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_bottom_left.data(), ghost_to_bottom_left_count, PARTICLE,
                  rank_bottom_left, 14, MPI_COMM_WORLD, &send_requests[6]);
    }
    if (rank_bottom_right != MPI_PROC_NULL) {
        MPI_Isend(ghost_to_bottom_right.data(), ghost_to_bottom_right_count, PARTICLE,
                  rank_bottom_right, 15, MPI_COMM_WORLD, &send_requests[7]);
    }
    
    // std::cout << "Rank " << rank << " After all Isend calls of particle data" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << rank << " after sending particle datas" << std::endl;
    // MPI_Waitall(8, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(8, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(8, send_requests, MPI_STATUSES_IGNORE);

    // Wait for all sends/receives to complete

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
    }

    // ============================== MOVE PARTICLES ============================== //
    for (size_t i = 0; i < local_parts.size(); i++) {
        move(local_parts[i], size);
    }
// ============================== PARTICLE EXCHANGE ACROSS RANKS ================================= //

std::vector<particle_t> particles_to_left;
std::vector<particle_t> particles_to_right;
std::vector<particle_t> particles_to_above;
std::vector<particle_t> particles_to_below;
std::vector<particle_t> particles_to_top_left;
std::vector<particle_t> particles_to_top_right;
std::vector<particle_t> particles_to_bottom_left;
std::vector<particle_t> particles_to_bottom_right;

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

int num_particles_to[8] = {
    static_cast<int>(particles_to_left.size()),
    static_cast<int>(particles_to_right.size()),
    static_cast<int>(particles_to_above.size()),
    static_cast<int>(particles_to_below.size()),
    static_cast<int>(particles_to_top_left.size()),
    static_cast<int>(particles_to_top_right.size()),
    static_cast<int>(particles_to_bottom_left.size()),
    static_cast<int>(particles_to_bottom_right.size())
};

int num_particles_from[8] = {0, 0, 0, 0, 0, 0, 0, 0};

enum Direction {LEFT, RIGHT, ABOVE, BELOW, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT};

// Initialize neighbor ranks array and fill it
int neighbor_ranks[8] = {
    rank_left, rank_right, rank_above, rank_below,
    rank_top_left, rank_top_right, rank_bottom_left, rank_bottom_right
};

// MPI Requests 
MPI_Request particle_requests[16];
int request_count = 0;

for (int dir = 0; dir < 8; dir++) {
    if (neighbor_ranks[dir] != MPI_PROC_NULL) {
        // 송신 (Send) - 각 방향으로 보낼 파티클 수
        std::cout << "Rank " << rank << " sending " << num_particles_to[dir]
        << " particles to direction " << dir
        << " (neighbor rank " << neighbor_ranks[dir] << ")" << std::endl;
        MPI_Isend(&num_particles_to[dir], 1, MPI_INT, 
                  neighbor_ranks[dir], dir, MPI_COMM_WORLD, &particle_requests[request_count++]);

    
        // 수신 (Receive) - 각 방향에서 받을 파티클 수
        std::cout << "Rank " << rank << " expecting to receive " << num_particles_from[dir]
        << " particles from direction " << dir
        << " (neighbor rank " << neighbor_ranks[dir] << ")" << std::endl;
        MPI_Irecv(&num_particles_from[dir], 1, MPI_INT, 
                  neighbor_ranks[dir], dir, MPI_COMM_WORLD, &particle_requests[request_count++]);
        
    }
}
MPI_Waitall(request_count, particle_requests, MPI_STATUSES_IGNORE);

std::vector<particle_t> particles_from_left(num_particles_from[LEFT]);
std::vector<particle_t> particles_from_right(num_particles_from[RIGHT]);
std::vector<particle_t> particles_from_above(num_particles_from[ABOVE]);
std::vector<particle_t> particles_from_below(num_particles_from[BELOW]);
std::vector<particle_t> particles_from_top_left(num_particles_from[TOP_LEFT]);
std::vector<particle_t> particles_from_top_right(num_particles_from[TOP_RIGHT]);
std::vector<particle_t> particles_from_bottom_left(num_particles_from[BOTTOM_LEFT]);
std::vector<particle_t> particles_from_bottom_right(num_particles_from[BOTTOM_RIGHT]);

// Reset request count
request_count = 0;

// Send and receive actual particle data for all 8 directions
std::vector<particle_t> particles_from[8] = {
    particles_from_left,
    particles_from_right,
    particles_from_above,
    particles_from_below,
    particles_from_top_left,
    particles_from_top_right,
    particles_from_bottom_left,
    particles_from_bottom_right
};

std::vector<particle_t> particles_to[8] = {
    particles_to_left,
    particles_to_right,
    particles_to_above,
    particles_to_below,
    particles_to_top_left,
    particles_to_top_right,
    particles_to_bottom_left,
    particles_to_bottom_right
};

for (int dir = 0; dir < 8; dir++) {
    if (num_particles_to[dir] > 0 && neighbor_ranks[dir] != MPI_PROC_NULL) {
        MPI_Isend(particles_to[dir].data(), num_particles_to[dir], PARTICLE, 
                  neighbor_ranks[dir], dir + 8, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }
    if (num_particles_from[dir] > 0 && neighbor_ranks[dir] != MPI_PROC_NULL) {
        MPI_Irecv(particles_from[dir].data(), num_particles_from[dir], PARTICLE, 
                  neighbor_ranks[dir], dir + 8, MPI_COMM_WORLD, &particle_requests[request_count++]);
    }
}

// Wait for all non-blocking communication to complete
MPI_Waitall(request_count, particle_requests, MPI_STATUSES_IGNORE);


// ============================== INSERT RECEIVED PARTICLES ================================= //
// Horizontal
local_parts.insert(local_parts.end(), particles_from_left.begin(), particles_from_left.end());
local_parts.insert(local_parts.end(), particles_from_right.begin(), particles_from_right.end());
// Vertical
local_parts.insert(local_parts.end(), particles_from_above.begin(), particles_from_above.end());
local_parts.insert(local_parts.end(), particles_from_below.begin(), particles_from_below.end());
// Diagonal
local_parts.insert(local_parts.end(), particles_from_top_left.begin(), particles_from_top_left.end());
local_parts.insert(local_parts.end(), particles_from_top_right.begin(), particles_from_top_right.end());
local_parts.insert(local_parts.end(), particles_from_bottom_left.begin(), particles_from_bottom_left.end());
local_parts.insert(local_parts.end(), particles_from_bottom_right.begin(), particles_from_bottom_right.end());

    MPI_Barrier(MPI_COMM_WORLD);
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

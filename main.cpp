#define _USE_MATH_DEFINES
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <unordered_map>

using namespace std;
using namespace Eigen;

const float kB = 1.380649e-23f;
const float hbar = 1.054571817e-34f;
const float massAl = 4.48e-26f;
const float eVtoJ = 1.60218e-19f;
const float latticeConstant = 4.05e-10f;
const int gridSize = 4;
const int numAtoms = gridSize * gridSize * gridSize * 4;
const float De = 0.5f * eVtoJ;  
const float a_morse = 2.0e10f;  
const float r0 = latticeConstant / sqrt(2.0f) * 0.98f; 
const float cutoff = 1.2f * r0;
const float renderScale = 1.5f;
float temperature = 800.0f;
float normtemperature = 300.0f;
float cameraAngleX = 25.0f;
float cameraAngleY = -45.0f;
const float layerThreshold = 0.9f;
const float massAr = 6.63e-26f; 
const float sigma_Ar = 3.4e-10f; 
const float epsilon_Ar = 1.67e-21f; 
const float argonGasDensity = 0.0025f; 
const float meanArgonDistance = 33.0e-10; 
const int numArgon = 20; 
const float argonTemp = 200.0f; 
const float argonDensity = 2.0f; 
const float argonBoxSize = cbrt(numArgon / argonDensity) * 1e-9f; 
const float sigma_ArAl = 3.55e-10f;
const float main_speed = 2400.0f; 
const float randomness = 0.3f;
const float collisionDistance = 1e-10f; 
const float epsilon_ArAl = 0.008f * eVtoJ;

struct Atom {
    Vector3f position;
    Vector3f equilibrium;
    Vector3f velocity;
    Vector3f displacement;
    Vector3f force;
};
struct ArgonAtom {
    Vector3f position;
    Vector3f velocity;
    Vector3f force;
};
struct PhononMode {
    long double frequency;
    VectorXf eigenvector;
};
struct CellIndex {
    int x, y, z;
    bool operator==(const CellIndex& other) const {
        return tie(x, y, z) == tie(other.x, other.y, other.z);
    }
};
struct CellIndexHasher {
    size_t operator()(const CellIndex& ci) const {
        return hash<int>()(ci.x) ^ (hash<int>()(ci.y) << 1) ^ (hash<int>()(ci.z) << 2);
    }
};

vector<ArgonAtom> argonAtoms(numArgon);
const float argonLayerHeight = (gridSize + 2.0f) * latticeConstant; 
vector<Atom> atoms(numAtoms);
vector<vector<int>> neighborLists(numAtoms);
vector<PhononMode> phononModes;
unordered_map<CellIndex, vector<int>, CellIndexHasher> cellMap;
float cellSize = cutoff * 1.1f;



bool isTopLayerAtom(int atomId) {
    const float tolerance = 0.25 * latticeConstant;
    const float maxZ = (gridSize - 0.5f) * latticeConstant;
    return abs(atoms[atomId].position.y() - maxZ) < tolerance;
}


Vector3f directedVelocity(float temperature, float mass, float main_speed, float randomness) {
    random_device rd;
    mt19937 gen(rd());
    Vector3f base_velocity(0.0f, -main_speed, 0.0f);
    normal_distribution<float> dist(0.0f, sqrt(kB * temperature * randomness / mass));
    Vector3f random_component(dist(gen), dist(gen), dist(gen));
    return base_velocity + random_component;
}


void generateArgonGas() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> pos_dist(0.0f, argonBoxSize);
    Vector3f lattice_center(
        gridSize * latticeConstant * 0.5f,
        gridSize * latticeConstant * 0.5f,
        gridSize * latticeConstant * 0.5f
    );
    for (int i = 0; i < numArgon; ++i) {
        argonAtoms[i].position = Vector3f(
            pos_dist(gen),
            gridSize * latticeConstant + 30.0e-10f + pos_dist(gen), 
            pos_dist(gen)
        );
        argonAtoms[i].velocity = directedVelocity(argonTemp, massAr, main_speed, randomness);
        argonAtoms[i].force = Vector3f::Zero();
    }
}

void generateFCCLattice() {
    int index = 0;
    float a = latticeConstant;

    for (int x = 0; x < gridSize; ++x) {
         for (int y = 0; y < gridSize; ++y) {
            for (int z = 0; z < gridSize; ++z) {
                Vector3f base(x * a, y * a, z * a);
                atoms[index++] = {base, base, Vector3f::Zero(), Vector3f::Zero(), Vector3f::Zero()};
                atoms[index++] = {base + Vector3f(0.5f * a, 0.5f * a, 0), base + Vector3f(0.5f * a, 0.5f * a, 0), Vector3f::Zero(), Vector3f::Zero(), Vector3f::Zero()};
                atoms[index++] = {base + Vector3f(0.5f * a, 0, 0.5f * a), base + Vector3f(0.5f * a, 0, 0.5f * a), Vector3f::Zero(), Vector3f::Zero(), Vector3f::Zero()};
                atoms[index++] = {base + Vector3f(0, 0.5f * a, 0.5f * a), base + Vector3f(0, 0.5f * a, 0.5f * a), Vector3f::Zero(), Vector3f::Zero(), Vector3f::Zero()};
            }
         }
    }
}
void buildCellMap() {
    cellMap.clear();
    for (int i = 0; i < numAtoms; ++i) {
        CellIndex ci{
            static_cast<int>(floor(atoms[i].position.x() / cellSize)),
            static_cast<int>(floor(atoms[i].position.y() / cellSize)),
            static_cast<int>(floor(atoms[i].position.z() / cellSize))
        };
        cellMap[ci].push_back(i);
    }
}

void buildNeighborLists() {
    for (auto& list : neighborLists) {
        list.clear();
    }
    buildCellMap();
    for (const auto& cell_entry : cellMap) {
        const CellIndex& cell = cell_entry.first;
        const vector<int>& atoms_in_current_cell = cell_entry.second; 
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
               for (int dz = -1; dz <= 1; ++dz) {
                    CellIndex neighbor{cell.x + dx, cell.y + dy, cell.z + dz};
                    auto neighbor_it = cellMap.find(neighbor);
                    if (neighbor_it != cellMap.end()) {
                        for (int i : atoms_in_current_cell) {
                             for (int j : neighbor_it->second) {
                                if (i >= j) continue;  
                                Vector3f delta = atoms[j].position - atoms[i].position;
                                delta -= latticeConstant * Vector3f(
                                    round(delta.x() / (gridSize * latticeConstant)),
                                    round(delta.y() / (gridSize * latticeConstant)),
                                    round(delta.z() / (gridSize * latticeConstant)));
                                if (delta.norm() < cutoff) {
                                    neighborLists[i].push_back(j);
                                    neighborLists[j].push_back(i);
                                }
                             }
                        }
                    }
               }
            }
        }
    }
}


    Vector3f morseForce(const Vector3f & ri, const Vector3f & rj, bool isSurfaceI, bool isSurfaceJ) {
        Vector3f delta = rj - ri;
        float dist = delta.norm();
        if (dist < 1e-10f) return Vector3f::Zero();

        Vector3f dir = delta.normalized();
        float dr = dist - r0;

        float n = (temperature / normtemperature)+0.01;
        float f_harmonic = n * a_morse * De * (1 - exp(-a_morse * dr)) * exp(-a_morse * dr);

        
        float f_anharmonic = (isSurfaceI ? 0.5f : 0.2f) * (3.0f * dr * dr - 2.0f * dr * dr * dr);

        return (f_harmonic + f_anharmonic) * dir;
    }



    Vector3f safeLennardJonesForce(const Vector3f & pos1, const Vector3f & pos2,
        float sigma, float epsilon) 
    {
        Vector3f delta = pos2 - pos1;
        float dist = delta.norm();
        dist = max(dist, 0.7f * sigma); 
        if (dist > 2.5f * sigma) {
            float fade = 0.5f * (1.0f + cos(M_PI * (dist - 2.5f * sigma) / (0.5f * sigma)));
            epsilon *= fade;
        }
        float ratio = sigma / dist;
        float ratio6 = pow(ratio, 6);
        float ratio12 = pow(ratio, 12);
        if (ratio > 1.2f) {
            float softening = 0.5f * (1.0f + cos(M_PI * (ratio - 1.2f) / 0.3f));
            ratio12 *= softening;
            ratio6 *= softening;
        }
        return (24.0f * epsilon / dist) * (2.0f * ratio12 - ratio6) * delta.normalized();
    }



    void handleCollisions() {
        for (auto& argon : argonAtoms) {
            for (auto& al_atom : atoms) {
                Vector3f delta = argon.position - al_atom.position;
                float dist = delta.norm();

                if (dist < collisionDistance) {
                    Vector3f normal = delta.normalized();
                    Vector3f rel_velocity = argon.velocity - al_atom.velocity;
                    float velocity_along_normal = rel_velocity.dot(normal);
                    if (velocity_along_normal > 0) continue; 
                    float impulse = 2.0f * velocity_along_normal;
                    argon.velocity -= impulse * normal * 0.9f; 
                    float overlap = collisionDistance - dist;
                    argon.position += normal * overlap * 1.01f;
                }
            }
        }
    }
void computeHarmonicModes() {
    MatrixXf dynMatrix = MatrixXf::Zero(3 * numAtoms, 3 * numAtoms);
    const float k_harmonic = 2 * a_morse * a_morse * De;
    const float k_anharmonic = 0.1f * k_harmonic; 
    for (int i = 0; i < numAtoms; ++i) {
        for (int j : neighborLists[i]) {
            Vector3f r_ij = atoms[j].position - atoms[i].position;
            float dist = r_ij.norm();
            r_ij.normalize();    
            Matrix3f block_harmonic = k_harmonic * r_ij * r_ij.transpose();    
            float displacement_factor = (atoms[i].displacement - atoms[j].displacement).norm() / r0;
            Matrix3f block_anharmonic = k_anharmonic * displacement_factor * r_ij * r_ij.transpose();
            Matrix3f total_block = block_harmonic + block_anharmonic;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    dynMatrix(3 * i + a, 3 * j + b) = total_block(a, b);
                    dynMatrix(3 * i + a, 3 * i + b) -= total_block(a, b);
                }
            }
        }
    }
    dynMatrix /= massAl;
    SelfAdjointEigenSolver<MatrixXf> eigensolver(dynMatrix);
    phononModes.resize(3 * numAtoms);
    for (int i = 0; i < 3; ++i) {
        phononModes[i].frequency = 0.0;
        phononModes[i].eigenvector.setZero();
    }
    for (int i = 3; i < 3 * numAtoms; ++i) {
        phononModes[i].frequency = sqrt(abs(eigensolver.eigenvalues()[i])) / (2 * M_PI);
        phononModes[i].eigenvector = eigensolver.eigenvectors().col(i);
    }
}



void excitePhononModes() {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& atom : atoms) {
        atom.displacement = Vector3f::Zero();
    }
    for (int m = 3; m < 20; ++m) { 
        float omega = 2 * M_PI * phononModes[m].frequency;
        float hbar_omega = hbar * omega;
        float n_phonon = 1.0f / (exp((hbar_omega / (kB * temperature))) - 1.0f);
        float amplitude = sqrt((2 * n_phonon + 1) * hbar / (massAl * omega));
        for (int i = 0; i < numAtoms; ++i) {
            for (int a = 0; a < 3; ++a) {
                atoms[i].displacement[a] += amplitude * dist(gen) * phononModes[m].eigenvector[3 * i + a];
            }
        }
    }
}


Vector3f lennardJonesForce(const Vector3f& pos1, const Vector3f& pos2) {
    Vector3f delta = pos2 - pos1;
    float dist = delta.norm();
    if (dist > 3.0f * sigma_Ar) {
        return Vector3f::Zero();
    }
    Vector3f dir = delta.normalized();
    float ratio = sigma_Ar / dist;
    float ratio6 = pow(ratio, 6);
    float ratio12 = ratio6 * ratio6;
    return 24.0f * epsilon_Ar * (2.0f * ratio12 - ratio6) / dist * dir;
}

void computeForces() {
    for (auto& atom : atoms) atom.force = Vector3f::Zero();
    for (auto& argon : argonAtoms) argon.force = Vector3f::Zero();
#pragma omp parallel for
    for (int i = 0; i < numAtoms; ++i) {
        for (int j : neighborLists[i]) {
            if (i >= j) continue;
            Vector3f force = morseForce(atoms[i].position, atoms[j].position,
                isTopLayerAtom(i), isTopLayerAtom(j));
            atoms[i].force += force;
            atoms[j].force -= force;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < numArgon; ++i) {
        for (int j = i + 1; j < numArgon; ++j) {
            Vector3f force = lennardJonesForce(argonAtoms[i].position, argonAtoms[j].position);
            argonAtoms[i].force += force;
            argonAtoms[j].force -= force;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < numArgon; ++i) {
        for (int j = 0; j < numAtoms; ++j) {
            if (isTopLayerAtom(j)) { 
                Vector3f force = safeLennardJonesForce(argonAtoms[i].position, atoms[j].position, sigma_ArAl, epsilon_ArAl);
                argonAtoms[i].force += force;
                atoms[j].force -= force;
            }
        }
    }
}


void applyBottomLayerConstraints() {
    const float bottomThreshold = 0.25f * latticeConstant;
    for (auto& atom : atoms) {
        if (atom.position.y() < bottomThreshold) {
            atom.velocity.setZero();
            atom.force.setZero();
            atom.position = atom.equilibrium;
        }
    }
}

void integrateMotion(float dt) {
    for (auto& atom : atoms) {
        atom.velocity += 0.5f * (atom.force / massAl) * dt;
        atom.position += atom.velocity * dt;
    }

    for (auto& argon : argonAtoms) {
        argon.velocity += 0.5f * (argon.force / massAr) * dt;
        argon.position += argon.velocity * dt;
    }
    handleCollisions();
    computeForces();
    applyBottomLayerConstraints();
    for (auto& atom : atoms) {
        atom.velocity += 0.5f * (atom.force / massAl) * dt;
    }

    for (auto& argon : argonAtoms) {
        argon.velocity += 0.5f * (argon.force / massAr) * dt;
    }
    handleCollisions();
}

void drawCellBorders() {
    glColor3f(0.7f, 0.7f, 0.7f); 
    glLineWidth(1.0f);
    float systemWidth = gridSize * latticeConstant * renderScale;
    float a = latticeConstant * renderScale; 
    glBegin(GL_LINES);
    for (int x = 0; x < gridSize; x++) {
        for (int y = 0; y < gridSize; y++) {
            for (int z = 0; z < gridSize; z++) {
                Vector3f origin(x * a, y * a, z * a);
                glVertex3fv(origin.data());
                glVertex3f(origin.x() + a, origin.y(), origin.z());

                glVertex3f(origin.x() + a, origin.y(), origin.z());
                glVertex3f(origin.x() + a, origin.y() + a, origin.z());

                glVertex3f(origin.x() + a, origin.y() + a, origin.z());
                glVertex3f(origin.x(), origin.y() + a, origin.z());

                glVertex3f(origin.x(), origin.y() + a, origin.z());
                glVertex3fv(origin.data());

                glVertex3fv(origin.data());
                glVertex3f(origin.x(), origin.y(), origin.z() + a);

                glVertex3f(origin.x() + a, origin.y(), origin.z());
                glVertex3f(origin.x() + a, origin.y(), origin.z() + a);

                glVertex3f(origin.x() + a, origin.y() + a, origin.z());
                glVertex3f(origin.x() + a, origin.y() + a, origin.z() + a);

                glVertex3f(origin.x(), origin.y() + a, origin.z());
                glVertex3f(origin.x(), origin.y() + a, origin.z() + a);

                glVertex3f(origin.x(), origin.y(), origin.z() + a);
                glVertex3f(origin.x() + a, origin.y(), origin.z() + a);

                glVertex3f(origin.x() + a, origin.y(), origin.z() + a);
                glVertex3f(origin.x() + a, origin.y() + a, origin.z() + a);

                glVertex3f(origin.x() + a, origin.y() + a, origin.z() + a);
                glVertex3f(origin.x(), origin.y() + a, origin.z() + a);

                glVertex3f(origin.x(), origin.y() + a, origin.z() + a);
                glVertex3f(origin.x(), origin.y(), origin.z() + a);
            }
        }
    }
    glEnd();
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(a * 1.5, 0, 0);
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, a * 1.5, 0);
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, a * 1.5);
    glEnd();
}


void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float systemSize = gridSize * latticeConstant * renderScale;
    Vector3f center(systemSize * 0.5f, systemSize * 0.5f, systemSize * 0.5f);
    float cameraDistance = systemSize * 1.0f;
    float halfSize = systemSize * 0.7f;
    float aspectRatio = (float)glutGet(GLUT_WINDOW_WIDTH) / glutGet(GLUT_WINDOW_HEIGHT);
    float viewSize = systemSize * 1.0f; 
    if (aspectRatio > 1) {
        glOrtho(-viewSize, viewSize, -viewSize / aspectRatio, viewSize / aspectRatio, -viewSize * 2, viewSize * 2);
    }
    else {
        glOrtho(-viewSize * aspectRatio, viewSize * aspectRatio, -viewSize, viewSize, -viewSize * 2, viewSize * 2);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float camX = center.x() + cameraDistance * sin(cameraAngleY * M_PI / 180) * cos(cameraAngleX * M_PI / 180);
    float camY = center.y() + cameraDistance * sin(cameraAngleX * M_PI / 180);
    float camZ = center.z() + cameraDistance * cos(cameraAngleY * M_PI / 180) * cos(cameraAngleX * M_PI / 180);
    gluLookAt(camX, camY, camZ, center.x(), center.y(), center.z(), 0, 1, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawCellBorders();
    const float baseSize = systemSize * 0.01f;
    const float maxAnharmonicity = 0.5f; 
    for (const auto& atom : atoms) {
        glPushMatrix();
        Vector3f pos = atom.position * renderScale;
        glTranslatef(pos.x(), pos.y(), pos.z());
        float displacement = (atom.position - atom.equilibrium).norm();
        float anharmonicity = min(displacement / r0, maxAnharmonicity);
        float normalizedAnharm = anharmonicity / maxAnharmonicity;
        Vector3f color;
        if (normalizedAnharm < 0.5f) {
            color = Vector3f(0.0f, 2.0f * normalizedAnharm, 1.0f - 2.0f * normalizedAnharm);
        }
        else {
            color = Vector3f(2.0f * (normalizedAnharm - 0.5f), 1.0f - 2.0f * (normalizedAnharm - 0.5f), 0.0f);
        }
        glColor3f(color.x(), color.y(), color.z());
        float atomSize = baseSize * (1.0f + 0.5f * normalizedAnharm);
        glutSolidSphere(atomSize, 10, 10);
        if (normalizedAnharm > 0.7f) {
            float pulse = 0.1f * sin(glutGet(GLUT_ELAPSED_TIME) * 0.005f) + 0.9f;
            glColor4f(1.0f, 0.3f, 0.0f, 0.3f * pulse);
            glutSolidSphere(atomSize * 1.5f, 12, 12);
        }
        glPopMatrix();
    }
    const float arSize = latticeConstant * renderScale * 0.1f;
    for (const auto& argon : argonAtoms) {
        glPushMatrix();
        Vector3f pos = argon.position * renderScale;
        glTranslatef(pos.x(), pos.y(), pos.z());
        glColor3f(1.0f, 1.0f, 1.0f);
        glutSolidSphere(arSize, 10, 10);
        glPopMatrix();
    }
    glutSwapBuffers();
}

void update(int value) {
    static float time = 0;
    float dt = 2e-14f;
    time += dt;
    buildNeighborLists();
    integrateMotion(dt);
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}




void mouseMotion(int x, int y) {
    static int lastX = x, lastY = y;
    cameraAngleY += (x - lastX) * 0.5f;
    cameraAngleX += (y - lastY) * 0.5f;
    cameraAngleX = max(-89.0f, min(89.0f, cameraAngleX));
    lastX = x;
    lastY = y;
    glutPostRedisplay();
}
void printSimulationInfo() {
    std::cout << "\n=== Параметры симуляции ===\n";
    std::cout << "Тип решетки: ГЦК (FCC)\n";
    std::cout << "Температура решётки: " << temperature << " K\n";
    std::cout << "Температура аргона: " << argonTemp << " K\n";
    std::cout << "Направленная скорость аргона: " << main_speed << " K\n";
    std::cout << "Шаг по времени: " << 2e-14 << " с\n";
    std::cout << "Параметр решетки (a): " << latticeConstant << " м\n";
    std::cout << "Число атомов: " << numAtoms << "\n";
    std::cout << "Размер системы: " << gridSize << "x" << gridSize << "x" << gridSize << " ячеек\n";
    std::cout << "Граничные условия:\n";
    std::cout << "  - X,Y: периодические (PBC)\n";
    std::cout << "  - Z: свободная поверхность (для верхнего слоя)\n";
}
void printForceAndPotentialInfo() {
    std::cout << "\n=== Физические модели ===\n";
    std::cout << "Потенциал взаимодействия: Морзе\n";
    std::cout << "  U(r) = D_e * [1 - exp(-a*(r-r0))]^2\n";
    std::cout << "  Параметры:\n";
    std::cout << "    D_e = " << De << " Дж (" << De / eVtoJ << " эВ)\n";
    std::cout << "    a = " << a_morse << " м^-1\n";
    std::cout << "    r0 = " << r0 << " м\n";
    std::cout << "Сила:\n";
    std::cout << "  F(r) = 2*a*D_e*exp(-a*(r-r0))*[1-exp(-a*(r-r0))] * r\n";
    std::cout << "Ангармонические поправки:\n";
    std::cout << "  F_anharm = alpha*(3*delta_r^2 - 2*delta_r^3)\n";
    std::cout << "  alpha (объем) = 1.0, alpha (поверхность) = 2.0\n";
}




int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Эффекты проникновения молекул в кристаллическое тело ");

    printSimulationInfo();
    printForceAndPotentialInfo();

    generateFCCLattice();
    generateArgonGas();

    buildNeighborLists();
    computeHarmonicModes();
    excitePhononModes();

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glutDisplayFunc(display);
    glutMotionFunc(mouseMotion);
    glutTimerFunc(0, update, 0);
    glutMainLoop();
    return 0;
}




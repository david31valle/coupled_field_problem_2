# C++ FEM Framework — Ubuntu CLI Quick Start

Follow these steps **exactly** to build and run the project on Ubuntu from the command line.  
You received the project as a **ZIP via Dropbox**. Name the folder accordingly.

---

## 1) Install required tools (one-time)

```bash
sudo apt update
sudo apt install -y build-essential
# Install a recent CMake (meets CMakeLists.txt requirement >= 3.26)
sudo snap install cmake --classic
```

> If `snap` isn’t available, enable it first:
> ```bash
> sudo apt install -y snapd
> sudo snap install core
> sudo snap refresh
> sudo snap install cmake --classic
> ```

---

## 2) Unzip and open the project folder

1) Unzip the ZIP you downloaded from Dropbox.
2) Go to `config.ini` and change the path to `data.txt` as per your directory. For eg. `/home/project-folder/data.txt`
3) Open **Terminal** and go to the project folder — the one that contains `CMakeLists.txt`:
```bash
cd /path/to/your/project-folder
```

---

## 3) Configure and build (out-of-source)

```bash
mkdir -p output
cmake -S . -B output -DCMAKE_BUILD_TYPE=Release
cmake --build output -j"$(nproc)"
```

This produces the executable inside the `output/` directory. This may take a while.

---

## 4) Run the program

```bash
cd/output
./coupled_field_problem_2
```

- Make changes to parameters in the `config.ini` file before running the executable.
- If you changed the target name in `CMakeLists.txt`, run that executable instead. In this case, "coupled_field_problem_2".
- After the run is complete, the VTK output files can be then found in `vtk_output`.
- If you need to run again with different sets of parameters, save the vtk_output files in another directory and delete them before starting the new run. 

---

## 5) Clean rebuild (only if needed)

If a build fails, or you want to start fresh:
```bash 
cd ..
rm -rf output
mkdir -p output
cmake -S . -B output -DCMAKE_BUILD_TYPE=Release
cmake --build output -j"$(nproc)"
```

---

## Notes

- Make sure you are **inside the project folder** (where `CMakeLists.txt` is) before running the commands above.
- The code expects data paths used in `main.cpp` to be valid on your system. If you see a path error (e.g., a Windows path), change it to a valid local/relative path in `main.cpp` and rebuild (Steps 3–4).

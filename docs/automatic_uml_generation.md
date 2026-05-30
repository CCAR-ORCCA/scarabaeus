# Automatic UML Diagram Generation with pyreverse in the terminal



### Requirements
- **Pylint**: Supplies the main UML generator function, pyreverse. (Install with `pip install pylint`)
- **Graphviz**: Supplies the dot command.

### Installation Guide for GraphViz
- **For Windows**: 
   - Download and install GraphViz.exe [here](https://graphviz.org/download/) and add the path of Graphviz bin directory to the system path.

   **Verify the installation by running `dot -V` command in the terminal** 
   
   - Download and install Pygraphviz [here](https://pygraphviz.github.io/documentation/stable/install.html#install) and add the path of   PyGraphviz bin directory to the system path during installation.

   - Install "graphviz2drawio" with `pip install graphviz2drawio` 

- **For MacOS**: Install GraphViz using Homebrew.(Install with `brew install graphviz`)
  
**Verify the installation by running `dot -V` command in the terminal** 

### Output
- **Classes**: Outlines the actual dependencies on object usage.
- **Packages**: Provides pathways of import relationships.

## Steps to Create a Class Diagram

1. **Make sure all subfolders have an "empty" `__init__.py` file**, so as to identify them as part of the whole package.

2. **Navigate to the parent directory of your package where the code is stored.**  
   For us, that is `src/`.

3. **Run the command**:
   ```sh
   pyreverse scarabaeus
   ```
4. **Go to `scarabaeus/`** 
   ```sh
   cd scarabaeus
   ```
5. **Run the class_connector_uml.py**
   ```sh
   python -c "import class_connector_uml; class_connector_uml.edit_dot_file()"
   ```
6. **Go to `src/`**
   ```sh
   cd ..
   ``` 
7. **Run the command - To get an PNG image**
   ```sh
   dot -Tpng classes.dot -o auto_generated_uml.png
   ```
8. **Run the command - To get XML to use in drawio(Interactive GUI Editor)**
   ```sh
   graphviz2drawio classes.dot -o auto_generated_editable_uml.xml
   ```

9. **Look at the outputs generated** 
- auto_generated_uml.png
- auto_generated_editable_uml.xml (To use in Draw.io, select File/Import From - select the xml file)


NOTE: The green connection represents the constructor-level relationship, and the red connection represents the method-level (function-level) relationship between classes.
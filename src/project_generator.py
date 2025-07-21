import logging
from pathlib import Path
import os
import json
import re

logger = logging.getLogger(__name__)

class ProjectGenerator:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def generate_from_lld(self, lld_path: str, output_path: str, reference_project: str):
        """Generate a new C# project based on LLD and reference project patterns"""
        # Read LLD
        lld_content = Path(lld_path).read_text()
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze LLD and extract components (use global RAG context)
        components = self._extract_components_from_lld(lld_content)
        
        # Generate project structure
        self._generate_project_structure(output_dir, components)
        
        # Generate code for each component using global RAG context
        for component in components:
            self._generate_component_code(component, output_dir)
        
        # Generate project files
        self._generate_project_files(output_dir, components)
        
        logger.info(f"Project generated successfully at: {output_path}")
    
    def _extract_components_from_lld(self, lld_content: str) -> list:
        """Extract components from LLD using global RAG context"""
        question = f"""Based on this LLD document, extract all components that need to be generated:
        
{lld_content}

For each component, provide:
1. Component name
2. Type (namespace/class/interface/enum)
3. File path structure
4. Dependencies
5. Key methods and properties

Format the response as a structured list."""
        
        response = self.rag.query_global(question)
        
        # DEBUG: Print the raw LLM response for diagnosis
        print("\n--- RAW LLM COMPONENT EXTRACTION RESPONSE ---\n")
        print(response)
        print("\n--- END RAW LLM RESPONSE ---\n")
        
        # Parse response to extract components
        # In real implementation, this would parse the structured response
        components = self._parse_component_list(response)
        return components
    
    def _generate_project_structure(self, output_dir: Path, components: list):
        """Create directory structure for the project"""
        directories = set()
        
        for component in components:
            file_path = component.get('file_path', '')
            if '/' in file_path:
                directory = output_dir / Path(file_path).parent
                directories.add(directory)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _generate_component_code(self, component: dict, output_dir: Path):
        """Generate code for a single component using global RAG context"""
        component_type = component.get('type', 'class')
        component_name = component.get('name', 'Unknown')
        prompt = f"""Generate complete C# code for the following component based on the existing project patterns:\n\nComponent Type: {component_type}\nComponent Name: {component_name}\nDependencies: {component.get('dependencies', [])}\nDescription: {component.get('description', '')}\n\nRequirements:\n1. Follow the coding patterns and conventions from the reference projects\n2. Include all necessary using statements\n3. Implement all specified methods and properties\n4. Add appropriate XML documentation\n5. Follow the same error handling patterns\n6. Use the same naming conventions\n\nGenerate production-ready code."""
        code = self.rag.query_global(prompt)
        code = self._clean_generated_code(code)
        # Strip backticks from file path
        file_path = component.get('file_path', f"{component_name}.cs").replace('`', '')
        file_path = file_path.strip()
        file_path = file_path.lstrip('/')
        file_path = file_path.replace('..', '')  # Prevent directory traversal
        file_path = output_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code)
        logger.info(f"Generated: {file_path}")
    
    def _generate_project_files(self, output_dir: Path, components: list):
        """Generate .csproj and other project files using global RAG context"""
        # Generate .csproj
        question = """Based on the reference project, what should be included in the .csproj file?
        Extract: target framework, package references, project references, and other configurations."""
        
        project_info = self.rag.query_global(question)
        
        csproj_content = self._generate_csproj(project_info, components)
        (output_dir / "Generated.csproj").write_text(csproj_content)
        
        # Generate solution file
        sln_content = self._generate_solution_file("Generated", output_dir)
        (output_dir / "Generated.sln").write_text(sln_content)
    
    def _parse_component_list(self, response: str) -> list:
        """Robustly parse component list from LLM response, handling Markdown and alternate keys."""
        import re
        components = []
        current_component = {}
        lines = response.split('\n')
        def clean(line):
            # Remove Markdown, bullets, numbers, and extra whitespace
            line = re.sub(r'^\s*[-*+\d.]+\s*', '', line)
            line = re.sub(r'\*\*', '', line)
            line = line.strip()
            return line
        for line in lines:
            cline = clean(line)
            if not cline:
                continue
            # Component name (match 'Component:' or 'Component Name:')
            if re.match(r'(?i)component( name)?:', cline):
                if current_component.get('name') and current_component.get('file_path'):
                    components.append(current_component)
                current_component = {}
                current_component['name'] = cline.split(':', 1)[1].strip()
            # Type
            elif re.match(r'(?i)type:', cline):
                current_component['type'] = cline.split(':', 1)[1].strip().lower()
            # File path
            elif re.match(r'(?i)(file path structure|path|file):', cline):
                current_component['file_path'] = cline.split(':', 1)[1].strip()
            # Dependencies
            elif re.match(r'(?i)dependencies:', cline):
                deps = cline.split(':', 1)[1].strip()
                current_component['dependencies'] = [d.strip() for d in re.split(r',|;', deps) if d.strip()]
            # Methods
            elif re.match(r'(?i)(key methods and properties|methods):', cline):
                methods = cline.split(':', 1)[1].strip()
                current_component['methods'] = methods
            # Fields
            elif re.match(r'(?i)fields:', cline):
                fields = cline.split(':', 1)[1].strip()
                current_component['fields'] = fields
            # Constructor
            elif re.match(r'(?i)constructor:', cline):
                constructor = cline.split(':', 1)[1].strip()
                current_component['constructor'] = constructor
        if current_component.get('name') and current_component.get('file_path'):
            components.append(current_component)
        # DEBUG: Print parsed components
        print("\n--- PARSED COMPONENTS ---\n")
        for comp in components:
            print(comp)
        print("\n--- END PARSED COMPONENTS ---\n")
        return components
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and format generated code"""
        # Remove markdown code blocks if present
        if '```csharp' in code:
            code = code.split('```csharp')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1].split('```')[0]
        
        # Ensure proper line endings
        code = code.strip()
        
        return code
    
    def _generate_csproj(self, project_info: str, components: list) -> str:
        """Generate .csproj file content"""
        return """<Project Sdk=\"Microsoft.NET.Sdk\">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>

  <ItemGroup>
    <!-- Add package references based on analysis -->
  </ItemGroup>
</Project>"""
    
    def _generate_solution_file(self, project_name: str, project_dir: Path) -> str:
        """Generate .sln file content"""
        import uuid
        project_guid = str(uuid.uuid4()).upper()
        
        return f"""Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project(\"{{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}}\") = \"{project_name}\", \"{project_dir.name}\\{project_name}.csproj\", \"{{{project_guid}}}\"
EndProject
Global
    GlobalSection(SolutionConfigurationPlatforms) = preSolution
        Debug|Any CPU = Debug|Any CPU
        Release|Any CPU = Release|Any CPU
    EndGlobalSection
    GlobalSection(ProjectConfigurationPlatforms) = postSolution
        {{{project_guid}}}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
        {{{project_guid}}}.Debug|Any CPU.Build.0 = Debug|Any CPU
        {{{project_guid}}}.Release|Any CPU.ActiveCfg = Release|Any CPU
        {{{project_guid}}}.Release|Any CPU.Build.0 = Release|Any CPU
    EndGlobalSection
EndGlobal""" 
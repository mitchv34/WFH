import pandas as pd
import re
import logging
from typing import Dict, List, Optional, Union
import os

class OccupationCode:
    def __init__(self, code: str, title: str, level: Optional[str] = None):
        self.code = code
        self.title = title
        self.parent_code = None
        if level is None:
            self.level = self._determine_level(code)
        else:
            self.level = level
    
    def _determine_level(self, code: str) -> str:
        """Determine the hierarchical level based on the code pattern."""
        if '.' in code:
            return 'onet'
        elif code.endswith('0000'):
            return 'major'
        elif code.endswith('00'):
            return 'minor'
        elif code.endswith('0'):
            return 'broad'
        else:
            return 'detailed'
    
    def __str__(self) -> str:
        return f"{self.code}: {self.title} ({self.level})"


class OccupationHierarchy:
    def __init__(self):
        self.codes: Dict[str, OccupationCode] = {}
        # Maps level to relevant length of code prefix for finding parents
        self.level_prefixes = {
            'detailed': 5,  # First 5 chars like "11-30" to find broad parent
            'broad': 4,     # First 4 chars like "11-3" to find minor parent
            'minor': 2      # First 2 chars like "11" to find major parent
        }
        # Load croswalks by default
        ## Load SOC 2018 structure
        try: 
            self.load_soc_2018()
        except Exception as e:
            print(f"Warning: Could not load SOC 2018 structure - {e}")
            self.codes = {}
        ## Load ONET SOC crosswalk if available
        try:
            self.load_onet_soc_crosswalk()
        except Exception as e:
            self.onet_soc_crosswalk = None
            print(f"Warning: Could not load ONET SOC crosswalk - {e}")
        

        
    def load_soc_2018(self, filepath: str = "data/aux_and_croswalks/soc_structure_2018.csv") -> None:
        """
        Load SOC 2018 structure from the specified CSV file.
        Default path is data/aux_and_croswalks/soc_structure_2018.csv
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Error: File not found at {filepath}")
                return
                
            # Read CSV file with pandas
            df = pd.read_csv(filepath)
            
            # Process the dataframe based on SOC 2018 structure
            self._process_soc_structure(df)
            
            # After loading all codes, establish parent-child relationships
            self._build_relationships()
            
            print(f"Successfully loaded {len(self.codes)} occupation codes from {filepath}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _process_soc_structure(self, df: pd.DataFrame) -> None:
        """Process the SOC structure dataframe."""
        # Expected columns for SOC 2018 structure
        expected_columns = ['Major Group', 'Minor Group', 'Broad Group', 'Detailed Occupation']
        
        # Check if the dataframe has the expected structure
        for col in expected_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in the dataframe")
        
        # Process each row of the DataFrame
        for _, row in df.iterrows():
            # Process Major Group
            if pd.notna(row.get('Major Group', '')):
                code = str(row['Major Group']).strip()
                if code.startswith('15-'):
                    a = 1
                # Find the title in the last non-empty column
                title = None
                for col in reversed(df.columns):
                    if pd.notna(row.get(col, '')) and col not in expected_columns:
                        title = str(row[col]).strip()
                        if title:  # Check if title is not empty
                            break
                
                if code and title:
                    self.codes[code] = OccupationCode(code, title, level='major')
            
            # Process Minor Group
            if pd.notna(row.get('Minor Group', '')):
                code = str(row['Minor Group']).strip()
                if code.startswith('15-'):
                    a = 1
                # Find the title in the last non-empty column
                title = None
                for col in reversed(df.columns):
                    if pd.notna(row.get(col, '')) and col not in expected_columns:
                        title = str(row[col]).strip()
                        if title:  # Check if title is not empty
                            break
                
                if code and title:
                    self.codes[code] = OccupationCode(code, title, level='minor')
            
            # Process Broad Group
            if pd.notna(row.get('Broad Group', '')):
                code = str(row['Broad Group']).strip()
                if code.startswith('15-'):
                    a = 1
                # Find the title in the last non-empty column
                title = None
                for col in reversed(df.columns):
                    if pd.notna(row.get(col, '')) and col not in expected_columns:
                        title = str(row[col]).strip()
                        if title:  # Check if title is not empty
                            break
                
                if code and title:
                    self.codes[code] = OccupationCode(code, title, level='broad')
            
            # Process Detailed Occupation
            if pd.notna(row.get('Detailed Occupation', '')):
                code = str(row['Detailed Occupation']).strip()
                if code.startswith('15-'):
                    a = 1
                # Find the title in the last non-empty column
                title = None
                for col in reversed(df.columns):
                    if pd.notna(row.get(col, '')) and col not in expected_columns:
                        title = str(row[col]).strip()
                        if title:  # Check if title is not empty
                            break
                
                if code and title:
                    self.codes[code] = OccupationCode(code, title, level='detailed')
    
    def _build_relationships(self) -> None:
        """Build parent-child relationships between codes."""
        for code, obj in self.codes.items():
            if obj.level in self.level_prefixes:
                # Determine prefix length based on level
                prefix_len = self.level_prefixes[obj.level]
                prefix = code[:prefix_len]
                
                # Find potential parent candidates
                for parent_code, parent_obj in self.codes.items():
                    # Check if this is a direct parent (one level up)
                    if parent_code.startswith(prefix) and parent_obj.level == self._get_parent_level(obj.level):
                        obj.parent_code = parent_code
                        break
    
    def _get_parent_level(self, level: str) -> Optional[str]:
        """Get the parent level for a given level."""
        if level == 'detailed':
            return 'broad'
        elif level == 'broad':
            return 'minor'
        elif level == 'minor':
            return 'major'
        return None
    
    def get_code_info(self, code: str) -> Dict:
        """Get complete information about a specific code."""
        if code not in self.codes:
            return {'error': 'Code not found'}
            
        occupation = self.codes[code]
        result = {
            'code': occupation.code,
            'title': occupation.title,
            'level': occupation.level
        }
        
        # Find ancestry - all parent codes up to the major level
        if occupation.level != 'major':
            ancestry = self.get_ancestry(code)
            result['ancestry'] = ancestry
        
        return result
    
    def get_ancestry(self, code: str) -> List[Dict]:
        """Get a list of all parent codes up to the major level."""
        ancestry = []
        current_code = code
        
        while current_code in self.codes:
            current = self.codes[current_code]
            if current.parent_code:
                parent = self.codes.get(current.parent_code)
                if parent:
                    ancestry.append({
                        'code': parent.code,
                        'title': parent.title,
                        'level': parent.level
                    })
                    current_code = parent.code
                    # Stop if we've reached the major level
                    if parent.level == 'major':
                        break
                else:
                    break
            else:
                break
        
        return ancestry
    
    def identify_level(self, code: str) -> str:
        """Identify the level of a given code."""
        if code in self.codes:
            return self.codes[code].level
        
        # If code isn't in our database, determine level by pattern
        dummy = OccupationCode(code, "")
        return dummy.level
    
    def get_parent(self, code: str, target_level: Optional[str] = None) -> Optional[OccupationCode]:
        """
        Get the parent of a code, optionally specifying the target parent level.
        If target_level is provided, returns the ancestor at that level.
        """
        if code not in self.codes:
            return None
            
        current = self.codes[code]
        
        # If no specific level requested, return immediate parent
        if not target_level:
            return self.codes.get(current.parent_code)
        
        # If specific level requested, traverse up to that level
        ancestry = self.get_ancestry(code)
        for ancestor in ancestry:
            if ancestor['level'] == target_level:
                return self.codes.get(ancestor['code'])
        
        return None
    
    def print_code_hierarchy(self, code: str) -> None:
        """Print the full hierarchy for a specific code."""
        if code not in self.codes:
            print(f"Code {code} not found")
            return
            
        info = self.get_code_info(code)
        print(f"\nHierarchy for {info['code']}: {info['title']} ({info['level']})")
        
        if 'ancestry' in info:
            print("Ancestry:")
            for ancestor in info['ancestry']:
                print(f"  {ancestor['level'].capitalize()}: {ancestor['code']}: {ancestor['title']}")
    
    def list_all_codes(self, level: Optional[str] = None) -> None:
        """List all codes, optionally filtered by level."""
        count = 0
        for code, obj in sorted(self.codes.items()):
            if level is None or obj.level == level:
                print(f"{code}: {obj.title} ({obj.level})")
                count += 1
        print(f"\nTotal: {count} codes")
    
    def load_onet_soc_crosswalk(self, crosswalk_path="data/aux_and_croswalks/onet_soc_croswalk.csv"):
        """
        Load the ONET SOC crosswalk to include mappings from detailed SOC codes to ONET codes.

        Parameters:
        -----------
        crosswalk_path : str
            Path to the ONET SOC crosswalk CSV file.
        """
        try:
            crosswalk_df = pd.read_csv(crosswalk_path)
            # Ensure the necessary columns exist
            if '2018 SOC Code' in crosswalk_df.columns and 'O*NET-SOC 2019 Code' in crosswalk_df.columns:
                self.onet_soc_crosswalk = crosswalk_df[['2018 SOC Code', 'O*NET-SOC 2019 Code', 'O*NET-SOC 2019 Title' ]]
                logging.info("ONET SOC crosswalk loaded successfully.")
            else:
                logging.error("The crosswalk file does not contain the required columns: '2018 SOC Code' and 'O*NET-SOC 2019 Code'.")
        except Exception as e:
            logging.error(f"Failed to load ONET SOC crosswalk: {e}")

    def map_onet_to_soc(self, onet_code):
        """
        Map an O*NET-SOC code (e.g., '11-1011.03') to its corresponding detailed SOC code (e.g., '11-1011')
        
        Parameters:
        -----------
        onet_code : str
            The ONET-SOC code to map
            
        Returns:
        --------
        str
            The corresponding detailed SOC code, or None if not found
        """
        # For O*NET-SOC codes, the first 7 characters typically match the detailed SOC code
        if onet_code and len(onet_code) > 7 and '.' in onet_code:
            # Extract the detailed SOC part (before the decimal)
            return onet_code.split('.')[0]
        
        # If the crosswalk is loaded, try to find the code there
        if hasattr(self, 'onet_soc_crosswalk'):
            matching_rows = self.onet_soc_crosswalk[self.onet_soc_crosswalk['O*NET-SOC 2019 Code'] == onet_code]
            if not matching_rows.empty:
                return matching_rows.iloc[0]['2018 SOC Code']
        
        return None
    def get_full_hierarchy(self, code: str, include_titles: bool = False) -> Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """
        Get the full hierarchy for a given code as a flat dictionary.
        
        Parameters:
        -----------
        code : str
            The code to get the hierarchy for. Can be a SOC code at any level or an O*NET code.
        include_titles : bool, optional
            If True, also returns titles for each occupation code. Default is False.
            
        Returns:
        --------
        If include_titles is False:
            Dict[str, List[str]]
                Dictionary with hierarchy levels as keys and lists of codes as values.
                For example:
                {
                    'major': ['11-0000'],
                    'minor': ['11-1000', '11-2000', ...],
                    'broad': ['11-1010', '11-2010', ...],
                    'detailed': ['11-1011', '11-2011', ...],
                    'onet': ['11-1011.00', '11-1011.03', ...]
                }
        If include_titles is True:
            Dict[str, Dict[str, List[str]]]
                Dictionary with 'codes' and 'titles' as keys, each containing dictionaries of levels and their respective codes/titles.
                For example:
                {
                    'codes': {
                        'major': ['11-0000'],
                        'minor': ['11-1000', '11-2000', ...],
                        ...
                    },
                    'titles': {
                        'major_title': ['Management Occupations'],
                        'minor_title': ['Top Executives', 'Advertising, Marketing, ...'],
                        ...
                    }
                }
        """
        hierarchy = {
            'major': [],
            'minor': [],
            'broad': [],
            'detailed': [],
            'onet': []
        }
        
        if include_titles:
            titles = {
                'major_title': [],
                'minor_title': [],
                'broad_title': [],
                'detailed_title': [],
                'onet_title': []
            }
        
        original_code = code
        is_onet_input = '.' in code
        
        # Handle ONET codes
        if is_onet_input:
            # This is an ONET code
            soc_code = self.map_onet_to_soc(code)
            if not soc_code or soc_code not in self.codes:
                return {'error': f'Could not map ONET code {code} to a valid SOC code'}
            # Store the original ONET code
            hierarchy['onet'] = [original_code]
            if include_titles:
                # Try to extract ONET title if available
                onet_title = "Unknown ONET Title"  # Default if title not found
                try:
                    onet_title = self.onet_soc_crosswalk.loc[self.onet_soc_crosswalk['O*NET-SOC 2019 Code'] == code, 'O*NET-SOC 2019 Title'].values[0]
                except IndexError:
                    print(f"Warning: ONET code {code} not found in crosswalk. Using default title.")
                titles['onet_title'] = [onet_title]
            # Use the mapped SOC code for the rest of the hierarchy
            code = soc_code
        
        # Check if the code exists in our database
        if code not in self.codes:
            return {'error': f'Code {code} not found in the SOC hierarchy'}
        
        # Determine the level of the code
        level = self.identify_level(code)
        
        # Add the code to its level in the hierarchy
        hierarchy[level].append(code)
        if include_titles:
            titles[f'{level}_title'].append(self.codes[code].title)
        
        # Get ancestors (for levels above the current code's level)
        if level != 'major':
            ancestry_list = self.get_ancestry(code)
            for ancestor in ancestry_list:
                ancestor_level = ancestor['level']
                hierarchy[ancestor_level].append(ancestor['code'])
                if include_titles:
                    titles[f'{ancestor_level}_title'].append(ancestor['title'])
        
        # Get descendants (for levels below the current code's level)
        # Find all codes that have this code in their ancestry
        for candidate_code, candidate_obj in self.codes.items():
            # Skip the code itself
            if candidate_code == code:
                continue
            
            # Skip codes at the same or higher level
            level_order = ['major', 'minor', 'broad', 'detailed']
            if level_order.index(candidate_obj.level) <= level_order.index(level):
                continue
            
            # Check if the current code is in the ancestry
            ancestry_list = self.get_ancestry(candidate_code)
            ancestry_codes = [a['code'] for a in ancestry_list]
            
            if code in ancestry_codes:
                hierarchy[candidate_obj.level].append(candidate_code)
                if include_titles:
                    titles[f'{candidate_obj.level}_title'].append(candidate_obj.title)
        
        # Only lookup other ONET codes if the input wasn't already an ONET code
        if hierarchy['detailed'] and not is_onet_input:
            if hasattr(self, 'onet_soc_crosswalk'):
                for detailed_code in hierarchy['detailed']:
                    matching_rows = self.onet_soc_crosswalk[self.onet_soc_crosswalk['2018 SOC Code'] == detailed_code]
                    if not matching_rows.empty:
                        onet_codes = matching_rows['O*NET-SOC 2019 Code'].tolist()
                        # Add to hierarchy
                        hierarchy['onet'].extend(onet_codes)
                        if include_titles:
                            # For simplicity, use the same title as the detailed occupation
                            detailed_title = next((self.codes[dc].title for dc in hierarchy['detailed'] if dc == detailed_code), "Unknown")
                            titles['onet_title'].extend([f"{detailed_title} ({oc})" for oc in onet_codes])
        
        # Remove empty levels
        hierarchy = {k: v for k, v in hierarchy.items() if v}
        if include_titles:
            titles = {k: v for k, v in titles.items() if v}
            return hierarchy, titles
        
        return hierarchy


# Example usage
if __name__ == "__main__":
    # Create the hierarchy object
    hierarchy = OccupationHierarchy()
        
    # Example 1: Get information for a detailed code and print its hierarchy
    print("\n--- Example 1: Detailed Code Information ---")
    # Let's use 11-1011 (Chief Executives) as an example
    hierarchy.print_code_hierarchy('11-1011')
    
    # Example 2: Map a detailed code to its major group
    print("\n--- Example 2: Finding Major Group for a Detailed Code ---")
    detailed_code = '11-3021'  # Computer and Information Systems Managers
    major_parent = hierarchy.get_parent(detailed_code, 'major')
    if major_parent:
        print(f"The major group for {detailed_code} is: {major_parent.code}: {major_parent.title}")
    
    # Example 3: List all codes at the major level
    print("\n--- Example 3: List All Major Groups (first 5) ---")
    count = 0
    for code, obj in sorted(hierarchy.codes.items()):
        if obj.level == 'major':
            print(f"{code}: {obj.title}")
            count += 1
            if count >= 5:  # Limit to first 5 for brevity
                print("...")
                break
    
    # Example 4: Map an O*NET-SOC code to its corresponding detailed SOC code
    print("\n--- Example 4: Map O*NET-SOC Code to Detailed SOC Code ---")
    onet_code = '11-1011.03'  # Chief Sustainability Officers
    soc_code = hierarchy.map_onet_to_soc(onet_code)
    if soc_code:
        print(f"The detailed SOC code for {onet_code} is: {soc_code}")
        
    # Example 5.1 given a O*NET code 
    print("\n--- Example 5.1: Full Hierarchy for O*NET Code ---")
    onet_code = '15-1211.00'  # Chief Sustainability Officers
    hierarchy_info = hierarchy.get_full_hierarchy(onet_code)
    print(hierarchy_info)
    # Example 5.2 given a SOC code minor 
    print("\n--- Example 5.2: Full Hierarchy for SOC Code ---")
    soc_code = '13-2000' # Financial Specialists
    hierarchy_info = hierarchy.get_full_hierarchy(soc_code)
    print(hierarchy_info)
    # Example 5.3 given a O*NET code return the full hierarchy with titles
    print("\n--- Example 5.3: Full Hierarchy with Titles ---")
    hierarchy_info, titles = hierarchy.get_full_hierarchy(onet_code, include_titles=True)
    print(titles)

    print("\n--- Usage Instructions ---")
    print("# 1. Create hierarchy object")
    print("hierarchy = OccupationHierarchy()")
    print("# 2. Load the SOC 2018 structure")
    print("hierarchy.load_soc_2018()")
    print("# 3. Identify the level of a code")
    print("level = hierarchy.identify_level('11-1011')  # 'detailed'")
    print("# 4. Get a code's parent at a specific level")
    print("major = hierarchy.get_parent('11-1011', 'major')  # Returns 11-0000 occupation")
    print("# 5. Print a code's full hierarchy")
    print("hierarchy.print_code_hierarchy('11-1011')")
    print("# 6. Map an O*NET-SOC code to its corresponding detailed SOC code")
    print("soc_code = hierarchy.map_onet_to_soc('11-1011.03')  # Returns '11-1011'")
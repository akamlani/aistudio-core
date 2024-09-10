import  pandas as pd 
from    typing import List 
from    pathlib import Path 
from    openpyxl.styles import PatternFill, Font

class ExcelFileWriter(object):
    """
    Example:
    filename_   = 'distribution-report.xlsx'
    sheet_name_ = 'admin'
    >>> xls    = ExcelFileWriter() 
    >>> df_xls = xls.read_excel(filename=filename_, sheet_name=sheet_name_).head()
    >>> xls.write_excel(dataset.data[['category', 'requested_by', 'volume']], filename_, sheet_name_, with_index=False)
    """
    def __init__(self, engine:str="openpyxl", **kwargs):
        self.engine = engine 
        self.kwargs = kwargs 
        # defaults: Create a pattern fill for coloring the header
        self.header_pattern = PatternFill(start_color="000000", end_color="000000", fill_type="solid") 
        self.text_color     = Font(color="FFFFFF")
        self.pad_width      = 5 


    def read_excel(self, filename:str, sheet_name:str|List[str], **kwargs) -> pd.DataFrame:
        with pd.ExcelFile(filename) as reader:  
            self.filename   = filename 
            self.sheet_name = sheet_name
            return pd.read_excel(reader, sheet_name, engine=self.engine, **kwargs)  


    def write_excel(self, df:pd.DataFrame, filename:str, sheet_name:str, with_index:bool=True, **kwargs):
        # if file exists, mode='a' or 'w'
        mappings = dict(mode='a', if_sheet_exists='replace') if Path(filename).exists() else dict(mode='w')

        with pd.ExcelWriter(
            filename, 
            engine=self.engine, 
            date_format="YYYY-MM-DD",
            datetime_format="YYYY-MM-DD",
            **mappings
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=with_index)
            # Access the XlsxWriter workbook and worksheet objects
            workbook   = writer.book
            sheets     = workbook.worksheets  
            worksheet  = writer.sheets[sheet_name]
            header_row = next(worksheet.iter_rows(min_row=1, max_row=1))
            # adjust header 
            self.adjust_header_cell(worksheet, header_row)
            self.adjust_widths(worksheet)

    def adjust_header_cell(self, worksheet, header, freeze_panes:bool=True) -> None:
        # Color the header row (assuming first row is the header)
        for cell in header:  
            cell.fill = self.header_pattern
            cell.font = self.text_color
        # Freeze the first row (A2 cell will be the top-left unfrozen cell), (above row 2)
        if freeze_panes:
            worksheet.freeze_panes = worksheet['A2']  

    def adjust_widths(self, worksheet, padding:int=5) -> None:
        # get column names 
        column_widths = {
            col[0].column_letter: max([len(str(cell.value)) if cell.value else 0 for cell in col]) + padding
            for col in worksheet.columns
        }
        # Set the column widths
        for col_letter, width in column_widths.items():
            worksheet.column_dimensions[col_letter].width = width

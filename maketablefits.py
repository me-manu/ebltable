# --- Imports -------- #
from collections import OrderedDict
from astropy.table import Column, Table
from astropy.io import fits
# -------------------- #

class TableCols(object):
    """
    Class to store columns for a fits table.
    Uses ordered dictionary, so insertion order matters!
    """

    def __init__(self):
	self.col_dict = OrderedDict()
	return 

    def add_col(self, name, **kwargs):
	"""
	Add a key to the dictionary that can be turned to fits table column.

	Parameters
	----------
	name: string, name of the column

	kwargs
	------
	Same as for astropy.table.Column function
	"""
	self.col_dict[name] = kwargs
	return 

    def makeTable(self):
	"""
	Create a table from the columns.

	Returns
	-------
	An astropy table object with the specified columns.
	"""
	cols = [Column(name=k, **v) for k,v in self.col_dict.items()]
	return Table(cols)

def table_to_fits_table(table):
    """
    Stolen from gammalib.utils, 
    http://docs.gammapy.org/en/latest/_modules/gammapy/utils/fits.html#table_to_fits_table
    Convert `~astropy.table.Table` to `astropy.io.fits.BinTableHDU`.

    The name of the table can be stored in the Table meta information
    under the ``name`` keyword.

    Parameters
    ----------
    table : `~astropy.table.Table`
	Table
    Returns
    -------
    hdu : `~astropy.io.fits.BinTableHDU`
	Binary table HDU
    """
    # read name and drop it from the meta information, otherwise
    # it would be stored as a header keyword in the BinTableHDU
    name = table.meta.pop('name', None)
    table.convert_unicode_to_bytestring(python3_only=True)
    data = table.as_array()
    header = fits.Header()
    header.update(table.meta)
    hdu = fits.BinTableHDU(data, header, name=name)
    # Copy over column meta-data
    for colname in table.colnames:
	if table[colname].unit is not None:
	    hdu.columns[colname].unit = table[colname].unit.to_string('fits')
    # TODO: this method works fine but the order of keywords in the table
    # header is not logical: for instance, list of keywords with column
    # units (TUNITi) is appended after the list of column keywords
    # (TTYPEi, TFORMi), instead of in between.
    # As a matter of fact, the units aren't yet in the header, but
    # only when calling the write method and opening the output file.
    # https://github.com/gammapy/gammapy/issues/298
    return hdu

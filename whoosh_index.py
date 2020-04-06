

import whoosh
from whoosh.index import create_in
from whoosh import fields
from whoosh import index
from import_data import import_data

def get_schema():
    return fields.Schema(
        title = fields.TEXT(stored = True), 
        id = fields.ID(stored = True), 
        content = fields.TEXT(stored = False), 
        authors = fields.KEYWORD(scorable=True, commas = True),
        tags = fields.KEYWORD(scorable=True),
    )

def create_index(index_dir):

    SCHEMA = get_schema()

    ix = index.create_in(index_dir, SCHEMA)
    writer = ix.writer()

    print('Loading data ...')
    data, sim = import_data('./dash_sample.csv')
    print('Data loaded')

    for row in data.reset_index().iterrows():
        row = row[1]
        writer.add_document(
            title = row.title,
            id = row.id,
            content = row.abstract,
            authors = ','.join(row.authors_list),
            tags = ' '.join(row.disease_tags),
        )
    writer.commit()

def get_index(index_dir):

    return index.open_dir(index_dir)


if __name__ == "__main__":

    create_index('./search_index')
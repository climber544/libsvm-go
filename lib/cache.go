package main

type cacheNode struct {
	data []float64
	next *cacheNode
	prev *cacheNode
}

type cache struct {
	head          []cacheNode
	lruHead       cacheNode
	colSize       int
	colCacheAvail int
}

func (c *cache) lruDelete(h *cacheNode) {
	h.prev.next = h.next
	h.next.prev = h.prev
}

func (c *cache) lruInsert(h *cacheNode) {
	h.next = &(c.lruHead)
	h.prev = c.lruHead.prev
	h.prev.next = h
	h.next.prev = h
}

func (c *cache) getData(i int) []float64 {
	var newData bool = true

	if c.head[i].data != nil {
		h := &(c.head[i])
		c.lruDelete(h)
		newData = false
	}

	if newData {
		// new data
		if c.colCacheAvail == 0 { // no more space in cache
			// free a column
			old := c.lruHead.next // oldest cache column

			old.data = nil // deallocate the slice it is storing
			c.lruDelete(old)

			c.colCacheAvail++
		}

		c.head[i].data = make([]float64, c.colSize)

		c.colCacheAvail--
	}

	h := &(c.head[i])
	c.lruInsert(h)

	return c.head[i].data
}

func NewCache(l, colSize, cacheSize int) cache {
	head := make([]cacheNode, l)
	for i := 0; i < l; i++ {
		head[i].data = nil
		head[i].next = nil
		head[i].prev = nil
	}

	c := cache{head: head, colSize: colSize, colCacheAvail: cacheSize}
	c.lruHead.next = nil
	c.lruHead.prev = nil

	return c
}

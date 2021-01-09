namespace Leetcode
{
    /// <summary>
    /// 并查集
    /// </summary>
    public class UnionFind
    {
        int[] _roots;
        int[] _rank;
        public int size;

        public UnionFind(int n)
        {
            _roots = new int[n];
            for (int i = 0; i < n; i++)
            {
                _roots[i] = i;
                _rank[i] = 1;
            }
            size = n;
        }

        public int Find(int i)
        {
            // if (i == _roots[i])
            // {
            //     return i;
            // }
            // else
            // {
            //     _roots[i] = Find(_roots[i]);
            //     return _roots[i];
            // }
            return i == _roots[i] ? i : (_roots[i] = Find(_roots[i]));
        }

        public void Union(int i, int j)
        {
            int x = Find(i), y = Find(j);
            if (_rank[x] <= _rank[y])
            {
                _roots[x] = y;
            }
            else
            {
                _roots[y] = x;
            }
            if (_rank[x] == _rank[y] && x != y)
            {
                _rank[y]++;
            }

            if (x != y)
            {
                size--;
            }
        }
    }
}
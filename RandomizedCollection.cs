using System;
using System.Linq;
using System.Collections.Generic;
/// <summary>
/// 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
/// </summary>
public class RandomizedCollection
{
    private List<int> _LIST;
    private int _LENGTH;
    Dictionary<int, HashSet<int>> _DICINDEX;
    /** Initialize your data structure here. */
    public RandomizedCollection()
    {
        _LIST = new List<int>();
        _LENGTH = 0;
        _DICINDEX = new Dictionary<int, HashSet<int>>();
    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public bool Insert(int val)
    {
        if (_LIST.Count > _LENGTH)
        {
            _LIST[_LENGTH] = val;
        }
        else
        {
            _LIST.Add(val);
        }
        _LENGTH++;
        bool isExists = false;
        if (_DICINDEX.ContainsKey(val))
        {
            isExists = _DICINDEX[val].Count > 0;
            _DICINDEX[val].Add(_LENGTH - 1);
        }
        else
        {
            _DICINDEX.Add(val, new HashSet<int>() { _LENGTH - 1 });
        }
        return !isExists;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public bool Remove(int val)
    {
        int index = -1;
        if (_DICINDEX.ContainsKey(val) && _DICINDEX[val].Count > 0)
        {
            index = _DICINDEX[val].FirstOrDefault();
            _DICINDEX[val].Remove(index);
        }
        if (index == -1) return false;
        if (index == _LENGTH - 1)
        {
            _LENGTH--;
            return true;
        }
        _DICINDEX[_LIST[index]].Remove(_LENGTH);
        _DICINDEX[_LIST[index]].Add(index);
        return true;
    }

    /** Get a random element from the collection. */
    public int GetRandom()
    {
        int res = -1;
        int seed = Guid.NewGuid().GetHashCode();
        Random rd = new Random(seed);
        int idx = rd.Next(_LENGTH);
        if (idx >= 0 && _LIST.Count > 0)
            res = _LIST[idx];
        return res;
    }
}

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection obj = new RandomizedCollection();
 * bool param_1 = obj.Insert(val);
 * bool param_2 = obj.Remove(val);
 * int param_3 = obj.GetRandom();
 */